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

func TestExtractAssetsLibsAndRemote(t *testing.T) {
	t.Setenv("LAYA_CACHE_DIR", t.TempDir())
	src := fstest.MapFS{
		"assets/llama/libllama.so.0.4.1":         {Data: []byte("stock")},
		"assets/llama-deberta/libllama.so.0.5.0": {Data: []byte("patched")},
		"assets/laya/tokenizer.json":             {Data: []byte("{}")},
		"assets/gliner2-decide/tokenizer.json":   {Data: []byte("{}")},
	}
	// GLiNER runs on the patched libs, the others on the stock ones.
	lib, _, err := extractAssets(src, "gliner2-decide")
	if err != nil || filepath.Base(lib) != "llama-deberta" {
		t.Fatalf("gliner libs: %s, %v", lib, err)
	}
	if _, err := os.Stat(filepath.Join(lib, "libllama.so.0")); err != nil {
		t.Error("symlink missing in patched libs")
	}
	lib, _, err = extractAssets(src, "laya")
	if err != nil || filepath.Base(lib) != "llama" {
		t.Fatalf("laya libs: %s, %v", lib, err)
	}
	if got := embeddedModels(src); len(got) != 2 {
		t.Errorf("lib dirs listed as models: %v", got)
	}

	// A downloadable model gets the stock libs and an (empty) dir under the cache root; no model files are extracted.
	lib, dir, err := extractAssets(src, "qwen3guard")
	if err != nil || filepath.Base(lib) != "llama" || filepath.Base(dir) != "qwen3guard" || filepath.Base(filepath.Dir(dir)) != "models" {
		t.Fatalf("qwen3guard: lib %s dir %s err %v", lib, dir, err)
	}
	// The policy chat model is not a classifier.
	if _, _, err := extractAssets(src, "policy"); err == nil {
		t.Error("policy accepted as LAYA_MODEL")
	}
}
