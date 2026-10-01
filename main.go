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

func run(log *slog.Logger) error {
	// Default to the embedded libs and model; LAYA_LLAMA_LIB / LAYA_DIR use external files instead.
	libDir, dir := os.Getenv("LAYA_LLAMA_LIB"), os.Getenv("LAYA_DIR")
	if libDir == "" || dir == "" {
		start := time.Now()
		embLib, embDir, err := extractAssets(assets)
		if err != nil {
			return fmt.Errorf("extract embedded assets: %w", err)
		}
		log.Info("embedded assets ready", "ms", time.Since(start).Milliseconds())
		libDir, dir = cmp.Or(libDir, embLib), cmp.Or(dir, embDir)
	}
	gguf := env("LAYA_GGUF", "laya-multilingual-F16.gguf")

	s := &server{
		log:     log,
		timeout: envDuration("LAYA_REQUEST_TIMEOUT", 30*time.Second),
		sem:     make(chan struct{}, envInt("LAYA_MAX_INFLIGHT", 16, 1024)),
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
	m, err := laya.Open(laya.Options{Dir: dir, GGUF: gguf, GPULayers: envInt("LAYA_GPU_LAYERS", 0, 1024), Threads: envInt("LAYA_THREADS", 0, 1024), Contexts: envInt("LAYA_CONTEXTS", 1, 64)})
	if err != nil {
		return err
	}
	s.model = m

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
		WriteTimeout:      s.timeout + 5*time.Second,
		IdleTimeout:       60 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

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
