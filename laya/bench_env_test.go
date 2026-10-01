package laya

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
)

// Benchmark fixtures. Model files come from LAYA_DIR, else the embedded-asset cache
// (~/.cache/laya-server/<id>/laya), else ../assets/laya. Benchmarks skip when missing.
var (
	benchOnce  sync.Once
	benchTok   *Tokenizer
	benchHead  *Head
	benchCfg   *HeadConfig
	benchSkip  string
	benchModel string
)

func benchModelDir() string {
	if d := os.Getenv("LAYA_DIR"); d != "" {
		return d
	}
	home, _ := os.UserCacheDir()
	if m, _ := filepath.Glob(filepath.Join(home, "laya-server", "*", "laya")); len(m) > 0 {
		return m[0]
	}
	return "../assets/laya"
}

func benchLlamaLib() string {
	if d := os.Getenv("LAYA_LLAMA_LIB"); d != "" {
		return d
	}
	home, _ := os.UserCacheDir()
	if m, _ := filepath.Glob(filepath.Join(home, "laya-server", "*", "llama")); len(m) > 0 {
		return m[0]
	}
	return ""
}

// benchSetup loads tokenizer and head once; it skips b when the files are absent.
func benchSetup(b *testing.B) (*Tokenizer, *Head, *HeadConfig) {
	b.Helper()
	benchOnce.Do(func() {
		benchModel = benchModelDir()
		var err error
		if benchTok, err = LoadTokenizer(filepath.Join(benchModel, "tokenizer.json")); err != nil {
			benchSkip = err.Error()
			return
		}
		if benchHead, benchCfg, err = LoadHead(filepath.Join(benchModel, "laya-multilingual-head.safetensors")); err != nil {
			benchSkip = err.Error()
		}
	})
	if benchSkip != "" {
		b.Skip("model files unavailable: ", benchSkip)
	}
	return benchTok, benchHead, benchCfg
}
