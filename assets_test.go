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
		"assets/llama/libllama.so.0.4.1":   {Data: []byte("lib")},
		"assets/llama/libggml-cuda.so":     {Data: []byte("x")},
		"assets/laya/tokenizer.json":       {Data: []byte("{}")},
		"assets/laya-guard/tokenizer.json": {Data: []byte("{}")},
		"assets/laya-guard/model.gguf":     {Data: []byte("g")},
	}
	lib, model, err := extractAssets(src, "laya")
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

	// the model that was not selected is not written
	if _, err := os.Stat(filepath.Join(filepath.Dir(model), "laya-guard")); err == nil {
		t.Error("unselected model was extracted")
	}

	// second run keeps existing files and is idempotent
	if _, _, err := extractAssets(src, "laya"); err != nil {
		t.Fatalf("second extract: %v", err)
	}

	// another embedded model, and bad names
	_, guard, err := extractAssets(src, "laya-guard")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(guard, "model.gguf")); err != nil {
		t.Errorf("laya-guard not extracted: %v", err)
	}
	for _, bad := range []string{"", "..", "../etc", "nope", "llama"} {
		if _, _, err := extractAssets(src, bad); err == nil {
			t.Errorf("extractAssets(%q) should fail", bad)
		}
	}
}
