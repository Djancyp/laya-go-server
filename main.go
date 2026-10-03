// Command laya-server serves the Laya classifier over HTTP.
//
// It exposes POST /v1/systemone in the TypeSafe System One request shape, answered
// locally by the Laya encoder (llama.cpp GGUF) and decision head.
package main

import (
	"cmp"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"laya-server/laya"
)

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stderr, nil))
	if err := run(log); err != nil {
		log.Error("fatal", "err", err)
		os.Exit(1)
	}
}

// defaultModel is the embedded model (a dir under assets/) that runs when LAYA_MODEL is unset.
const defaultModel = "gliner2-decide"

// defaultGGUF is the only *.gguf in dir (so a model dir needs no LAYA_GGUF), else the stock file name.
func defaultGGUF(dir string) string {
	if m, _ := filepath.Glob(filepath.Join(dir, "*.gguf")); len(m) == 1 {
		return filepath.Base(m[0])
	}
	return "laya-multilingual-F16.gguf"
}

func run(log *slog.Logger) error {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	// Default to the embedded libs and model; LAYA_LLAMA_LIB / LAYA_DIR use external files instead.
	libDir, dir := os.Getenv("LAYA_LLAMA_LIB"), os.Getenv("LAYA_DIR")
	if libDir == "" || dir == "" {
		start := time.Now()
		embLib, embDir, err := extractAssets(assets, env("LAYA_MODEL", defaultModel))
		if err != nil {
			return fmt.Errorf("extract embedded assets: %w", err)
		}
		log.Info("embedded assets ready", "ms", time.Since(start).Milliseconds())
		libDir, dir = cmp.Or(libDir, embLib), cmp.Or(dir, embDir)
		// A model too big to embed is downloaded on first start (and kept in the cache dir).
		if rm, ok := remoteModels[env("LAYA_MODEL", defaultModel)]; ok && os.Getenv("LAYA_DIR") == "" {
			if _, err := ensureRemote(ctx, log, rm, dir); err != nil {
				return err
			}
		}
	}
	gguf := cmp.Or(os.Getenv("LAYA_GGUF"), defaultGGUF(dir))

	s := &server{
		log:     log,
		timeout: envDuration("LAYA_REQUEST_TIMEOUT", 30*time.Second),
		sem:     make(chan struct{}, envInt("LAYA_MAX_INFLIGHT", 16, 1024)),
		// A first /v1/policy call converts every rule with the chat model (seconds, up to ~25 s for 48 rules).
		policySem:     make(chan struct{}, envInt("LAYA_POLICY_INFLIGHT", 2, 64)),
		policyTimeout: envDuration("LAYA_POLICY_TIMEOUT", 120*time.Second),
		limits: limits{
			MaxQuestions:    envInt("LAYA_MAX_QUESTIONS", 64, 1024),
			MaxStateBytes:   envInt("LAYA_MAX_STATE_BYTES", 64<<10, maxBodyBytes),
			MaxInstructions: 4 << 10,
		},
	}
	for k := range strings.SplitSeq(os.Getenv("LAYA_API_KEYS"), ",") {
		if k = strings.TrimSpace(k); k != "" {
			s.keys = append(s.keys, sha256.Sum256([]byte(k)))
		}
	}
	if len(s.keys) == 0 && os.Getenv("LAYA_ALLOW_NO_AUTH") != "true" {
		return errors.New("set LAYA_API_KEYS (comma separated), or LAYA_ALLOW_NO_AUTH=true to serve without auth")
	}

	if err := laya.InitLlama(libDir); err != nil {
		return err
	}
	m, err := laya.OpenAny(laya.Options{Dir: dir, GGUF: gguf, GPULayers: envInt("LAYA_GPU_LAYERS", 0, 1024), Threads: envInt("LAYA_THREADS", 0, 1024), Contexts: envInt("LAYA_CONTEXTS", 1, 64), MaxTokens: envInt("LAYA_MAX_TOKENS", 0, 8192)})
	if err != nil {
		return err
	}
	s.model = m
	s.startPolicy(ctx, log)

	// Fail fast on a broken model and pay the first-call cost before taking traffic.
	warm := []laya.Question{{ID: "warmup", Kind: laya.Noul, Instructions: "Is this a greeting?"}}
	if _, err := m.Predict("hello", warm); err != nil {
		return fmt.Errorf("warmup: %w", err)
	}
	s.ready.Store(true)

	srv := &http.Server{
		Addr:              env("LAYA_ADDR", ":8080"),
		Handler:           s.handler(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      max(s.timeout, s.policyTimeout) + 5*time.Second,
		IdleTimeout:       60 * time.Second,
	}

	errc := make(chan error, 1)
	go func() { errc <- srv.ListenAndServe() }()
	log.Info("listening", "addr", srv.Addr, "gguf", gguf, "auth", len(s.keys) > 0)

	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
	}

	log.Info("shutting down")
	s.ready.Store(false)
	shutCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutCtx); err != nil {
		log.Warn("shutdown", "err", err)
	}
	// Timed-out requests may still be inside Predict: take every slot so none is
	// running before the model is freed.
	for range cap(s.sem) {
		select {
		case s.sem <- struct{}{}:
		case <-shutCtx.Done():
			log.Warn("predictions still running; exiting without freeing the model")
			return nil
		}
	}
	m.Close()
	return nil
}

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

// envInt reads a positive int, falling back to def; values above max are clamped.
func envInt(k string, def, max int) int {
	if v, err := strconv.Atoi(os.Getenv(k)); err == nil && v > 0 {
		return min(v, max)
	}
	return def
}

func envDuration(k string, def time.Duration) time.Duration {
	if v, err := time.ParseDuration(os.Getenv(k)); err == nil && v > 0 {
		return v
	}
	return def
}

// startPolicy makes POST /v1/policy available: it loads the chat model from LAYA_POLICY_GGUF, or
// downloads the default one (Qwen3-0.6B Q8_0) in the background, so the server takes traffic meanwhile.
// LAYA_POLICY_MODEL=off turns the endpoint off.
func (s *server) startPolicy(ctx context.Context, log *slog.Logger) {
	if v := strings.ToLower(os.Getenv("LAYA_POLICY_MODEL")); v == "off" || v == "false" || v == "0" {
		return
	}
	s.policyLoading.Store(true)
	go func() {
		defer s.policyLoading.Store(false)
		path := os.Getenv("LAYA_POLICY_GGUF")
		if path == "" {
			dir, err := remoteDir(policyModelName)
			if err == nil {
				path, err = ensureRemote(ctx, log, remoteModels[policyModelName], dir)
			}
			if err != nil {
				log.Error("policy model unavailable: POST /v1/policy stays off", "err", err)
				return
			}
		}
		pc, err := laya.OpenPolicyCompiler(path, envInt("LAYA_GPU_LAYERS", 0, 1024), envInt("LAYA_THREADS", 0, 1024))
		if err != nil {
			log.Error("policy model failed to load: POST /v1/policy stays off", "err", err)
			return
		}
		s.setPolicy(pc)
		log.Info("policy model ready", "path", path)
		<-ctx.Done()
		s.setPolicy(nil)
		pc.Close()
	}()
}
