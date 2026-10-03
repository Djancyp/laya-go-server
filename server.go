package main

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"laya-server/laya"
)

// predictor is what the server needs from the model (a fake in tests).
type predictor interface {
	Predict(state string, qs []laya.Question) ([]laya.Result, error)
}

// moderator is implemented by moderation models (Qwen3Guard): a full assessment with categories
// and, for a response, the refusal flag.
type moderator interface {
	Moderate(prompt, response string) (laya.Moderation, error)
	// Answer answers questions from a prompt moderation, saving a second pass over the prompt.
	Answer(m laya.Moderation, qs []laya.Question) ([]laya.Result, error)
}

type policyHolder struct{ c policyCompiler }

func (s *server) setPolicy(c policyCompiler) {
	if c == nil {
		s.policyC.Store(nil)
		return
	}
	s.policyC.Store(&policyHolder{c})
}

type server struct {
	model predictor
	// policyC compiles policy text into questions; nil until the policy model is loaded.
	policyC       atomic.Pointer[policyHolder]
	policyLoading atomic.Bool // the policy model is being downloaded or loaded
	// policySem admits at most cap(policySem) /v1/policy requests: a compile takes seconds on the
	// compiler's single context, and must not use up the slots of /v1/systemone (sem).
	policySem     chan struct{}
	policyTimeout time.Duration
	log           *slog.Logger
	limits        limits
	timeout       time.Duration
	// sem admits at most cap(sem) requests, running or queued: the encoder runs one
	// sequence at a time, so a deeper queue only adds latency.
	sem   chan struct{}
	keys  [][32]byte // sha256 of each accepted API key
	ready atomic.Bool
}

const (
	maxBodyBytes = 1 << 20
	modelName    = "laya-multilingual"
)

func (s *server) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, _ *http.Request) {
		if !s.ready.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
		}
	})
	mux.Handle("POST /v1/systemone", s.auth(http.HandlerFunc(s.systemOne)))
	mux.Handle("POST /v1/policy", s.auth(http.HandlerFunc(s.policy)))
	return s.recoverer(s.logged(mux))
}

func (s *server) systemOne(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)
	var req request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			writeError(w, http.StatusRequestEntityTooLarge, "body_too_large", "request body over 1 MiB")
			return
		}
		writeError(w, http.StatusBadRequest, "bad_json", err.Error())
		return
	}
	mod, _ := s.model.(moderator)
	state, qs, err := req.parse(s.limits, mod != nil)
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, "invalid_request", err.Error())
		return
	}
	response, err := req.responseText(s.limits)
	if err == nil && response != "" && mod == nil {
		err = errors.New("response is only supported by moderation models")
	}
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, "invalid_request", err.Error())
		return
	}

	select {
	case s.sem <- struct{}{}:
	default:
		w.Header().Set("Retry-After", "1")
		writeError(w, 529, "overloaded", "too many requests in flight")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), s.timeout)
	defer cancel()

	type outcome struct {
		rs  []laya.Result
		mod *laya.Moderation
		err error
	}
	done := make(chan outcome, 1)
	go func() {
		// The slot is held until the model returns, even if the client gave up:
		// Predict cannot be cancelled, so freeing it early would let work pile up.
		defer func() { <-s.sem }()
		// recoverer only guards the handler goroutine; a panic here would kill the process.
		defer func() {
			if v := recover(); v != nil {
				s.log.Error("panic in predict", "value", v)
				done <- outcome{err: fmt.Errorf("panic: %v", v)}
			}
		}()
		var o outcome
		if mod != nil {
			m, err := mod.Moderate(state, response)
			o.mod, o.err = &m, err
		}
		if o.err == nil && len(qs) > 0 {
			if mod != nil && response == "" {
				o.rs, o.err = mod.Answer(*o.mod, qs)
			} else {
				o.rs, o.err = s.model.Predict(state, qs)
			}
		}
		done <- o
	}()

	select {
	case o := <-done:
		switch {
		case errors.Is(o.err, laya.ErrInput):
			writeError(w, http.StatusUnprocessableEntity, "invalid_request", o.err.Error())
		case o.err != nil:
			s.log.Error("predict failed", "err", o.err)
			writeError(w, http.StatusInternalServerError, "internal", "prediction failed")
		default:
			answers, u, err := toAnswers(qs, o.rs)
			if err != nil {
				s.log.Error("bad model result", "err", err)
				writeError(w, http.StatusInternalServerError, "internal", "prediction failed")
				break
			}
			body := map[string]any{"model": modelName, "answers": answers, "usage": u}
			if o.mod != nil {
				body["moderation"] = o.mod
				if response == "" {
					u.InputTokens = o.mod.Tokens // the questions reuse the moderation's pass
				} else {
					u.InputTokens += o.mod.Tokens
				}
				body["usage"] = u
			}
			writeJSON(w, http.StatusOK, body)
		}
	case <-ctx.Done():
		writeError(w, http.StatusGatewayTimeout, "timeout", "request timed out")
	}
}

// auth requires "Authorization: Bearer <key>" with a configured key.
func (s *server) auth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if len(s.keys) > 0 {
			tok, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
			if !ok || !s.validKey(tok) {
				w.Header().Set("WWW-Authenticate", `Bearer realm="laya"`)
				writeError(w, http.StatusUnauthorized, "unauthorized", "missing or invalid API key")
				return
			}
		}
		next.ServeHTTP(w, r)
	})
}

func (s *server) validKey(tok string) bool {
	sum := sha256.Sum256([]byte(tok))
	ok := 0
	for _, k := range s.keys { // no early exit: time does not reveal which key matched
		ok |= subtle.ConstantTimeCompare(sum[:], k[:])
	}
	return ok == 1
}

type statusWriter struct {
	http.ResponseWriter
	code int
}

// Unwrap lets http.ResponseController reach Flush and friends on the real writer.
func (w *statusWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }

func (w *statusWriter) WriteHeader(c int) {
	w.code = c
	w.ResponseWriter.WriteHeader(c)
}

func (s *server) logged(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-Id")
		if id == "" || len(id) > 64 {
			b := make([]byte, 8)
			_, _ = rand.Read(b)
			id = hex.EncodeToString(b)
		}
		w.Header().Set("X-Request-Id", id)

		start := time.Now()
		sw := &statusWriter{ResponseWriter: w, code: http.StatusOK}
		next.ServeHTTP(sw, r)
		if r.URL.Path == "/healthz" || r.URL.Path == "/readyz" {
			return // probes would drown the log
		}
		s.log.Info("request", "id", id, "method", r.Method, "path", r.URL.Path,
			"status", sw.code, "ms", time.Since(start).Milliseconds())
	})
}

func (s *server) recoverer(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if v := recover(); v != nil {
				if v == http.ErrAbortHandler {
					panic(v) // net/http's deliberate abort: let it through
				}
				s.log.Error("panic", "value", v, "path", r.URL.Path)
				writeError(w, http.StatusInternalServerError, "internal", "internal error")
			}
		}()
		next.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, code int, kind, msg string) {
	writeJSON(w, code, map[string]any{"error": map[string]string{"code": kind, "message": msg}})
}
