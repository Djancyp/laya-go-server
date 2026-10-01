package main

import (
	"os"
	"path/filepath"
	"testing"
	"testing/fstest"
)

func TestExtractAssets(t *testing.T) {
	t.Setenv("LAYA_CACHE_DIR", t.TempDir())
	src := fstest.MapFS{
		"assets/llama/libllama.so.0.4.1": {Data: []byte("lib")},
		"assets/llama/libggml-cuda.so":   {Data: []byte("x")},
		"assets/laya/tokenizer.json":     {Data: []byte("{}")},
	}
	lib, model, err := extractAssets(src)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{
		filepath.Join(lib, "libllama.so.0.4.1"), filepath.Join(lib, "libllama.so.0"), filepath.Join(lib, "libllama.so"),
		filepath.Join(model, "tokenizer.json"),
	} {
		if _, err := os.Stat(p); err != nil {
			t.Errorf("missing %s", p)
		}
	}

	// second run keeps existing files and is idempotent
	if _, _, err := extractAssets(src); err != nil {
		t.Fatalf("second extract: %v", err)
	}
}
