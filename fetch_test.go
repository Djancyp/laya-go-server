package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
)

func TestEnsureRemote(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		switch r.URL.Path {
		case "/org/repo/resolve/rev1/m.gguf":
			if r.Header.Get("Authorization") != "Bearer tok" {
				http.Error(w, "no", http.StatusUnauthorized)
				return
			}
			w.Write([]byte("model-bytes"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("HF_ENDPOINT", srv.URL)
	t.Setenv("HF_TOKEN", "tok")
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	dir := t.TempDir()
	sum := sha256.Sum256([]byte("model-bytes"))
	rm := remoteModel{"org/repo", "rev1", []remoteFile{{"m.gguf", 11, hex.EncodeToString(sum[:])}}}

	path, err := ensureRemote(context.Background(), log, rm, dir)
	if err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(path); string(b) != "model-bytes" || path != filepath.Join(dir, "m.gguf") {
		t.Errorf("got %q at %s", b, path)
	}
	if m, _ := filepath.Glob(filepath.Join(dir, "*.tmp")); len(m) != 0 {
		t.Errorf("temp files left: %v", m)
	}

	// Present: not downloaded again.
	before := hits.Load()
	if _, err := ensureRemote(context.Background(), log, rm, dir); err != nil || hits.Load() != before {
		t.Errorf("second call: err=%v, requests %d -> %d", err, before, hits.Load())
	}

	// A failed download leaves nothing behind.
	dir2 := t.TempDir()
	if _, err := ensureRemote(context.Background(), log, remoteModel{"org/repo", "rev1", []remoteFile{{"missing.gguf", 5, "00"}}}, dir2); err == nil {
		t.Error("404 accepted")
	}
	if m, _ := filepath.Glob(filepath.Join(dir2, "*")); len(m) != 0 {
		t.Errorf("left %v", m)
	}
}

func TestEnsureRemoteRejectsBadFiles(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("model-bytes")) }))
	defer srv.Close()
	t.Setenv("HF_ENDPOINT", srv.URL)
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	sum := sha256.Sum256([]byte("model-bytes"))
	good := hex.EncodeToString(sum[:])

	for name, f := range map[string]remoteFile{
		"wrong hash": {"m.gguf", 11, "00" + good[2:]},
		"wrong size": {"m.gguf", 12, good},
	} {
		dir := t.TempDir()
		if _, err := ensureRemote(context.Background(), log, remoteModel{"o/r", "rev", []remoteFile{f}}, dir); err == nil {
			t.Errorf("%s: accepted", name)
		}
		if m, _ := filepath.Glob(filepath.Join(dir, "*")); len(m) != 0 {
			t.Errorf("%s: left %v", name, m)
		}
	}

	// A file with the wrong size already in the cache is replaced, not trusted.
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "m.gguf"), []byte("truncated"), 0o644)
	if _, err := ensureRemote(context.Background(), log, remoteModel{"o/r", "rev", []remoteFile{{"m.gguf", 11, good}}}, dir); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(filepath.Join(dir, "m.gguf")); string(b) != "model-bytes" {
		t.Errorf("not replaced: %q", b)
	}
}

func TestCheckEndpoint(t *testing.T) {
	for ep, ok := range map[string]bool{
		"https://huggingface.co": true, "https://mirror.example.com/hf": true,
		"http://127.0.0.1:8080": true, "http://localhost:9": true,
		"http://mirror.example.com": false, "ftp://x": false, "huggingface.co": false,
	} {
		if err := checkEndpoint(ep); (err == nil) != ok {
			t.Errorf("checkEndpoint(%q) = %v", ep, err)
		}
	}
}
