package laya

import (
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

var (
	encOnce sync.Once
	encInst *Encoder
	encErr  error
)

// testEncoder loads the F16 GGUF (LAYA_GGUF overrides) on the CPU once. It skips when the model or the
// llama.cpp libraries are missing (LAYA_LLAMA_LIB overrides the kronk library dir).
func testEncoder(t *testing.T) *Encoder {
	t.Helper()
	gguf := modelDir + "/" + envOr("LAYA_GGUF", "laya-multilingual-F16.gguf")
	if _, err := os.Stat(gguf); err != nil {
		t.Skip("models/laya GGUF not downloaded")
	}
	lib := os.Getenv("LAYA_LLAMA_LIB")
	if lib == "" {
		home, _ := os.UserHomeDir()
		lib = filepath.Join(home, ".kronk", "libraries", "linux", "amd64", "cpu")
	}
	if _, err := os.Stat(lib); err != nil {
		t.Skip("llama.cpp libraries not found in ", lib)
	}

	encOnce.Do(func() {
		if encErr = InitLlama(lib); encErr == nil {
			encInst, encErr = NewEncoder(gguf, 1024, 0, 1, 0)
		}
	})
	if encErr != nil {
		t.Fatal(encErr)
	}
	return encInst
}

func cosine(a, b []float32) float64 {
	var ab, aa, bb float64
	for i := range a {
		ab += float64(a[i]) * float64(b[i])
		aa += float64(a[i]) * float64(a[i])
		bb += float64(b[i]) * float64(b[i])
	}
	return ab / math.Sqrt(aa*bb)
}

func TestEncoderMatchesHF(t *testing.T) {
	loadRef(t)
	enc := testEncoder(t)
	head, _ := testHead(t)

	for ci, c := range refCases {
		for qid, q := range c.PerQuestion {
			t.Run(qid+"/case"+string(rune('a'+ci)), func(t *testing.T) {
				want := q.hidden(t)
				got, err := enc.Hidden(q.IDs)
				if err != nil {
					t.Fatal(err)
				}
				if len(got) != len(want) {
					t.Fatalf("%d values, want %d", len(got), len(want))
				}

				// every token's hidden state must point the same way as PyTorch's
				worst := 1.0
				for i := 0; i < q.N; i++ {
					if cs := cosine(got[i*q.D:(i+1)*q.D], want[i*q.D:(i+1)*q.D]); cs < worst {
						worst = cs
					}
				}
				if worst < 0.99 {
					t.Errorf("worst per-token cosine %.4f, want >= 0.99", worst)
				}

				// and the head must reach (almost) the same logits from it
				logits, err := head.Logits(got, q.QType, q.Markers)
				if err != nil {
					t.Fatal(err)
				}
				for i := range logits {
					if d := math.Abs(float64(logits[i]) - q.Logits[i]); d > 0.5 {
						t.Errorf("logit %d = %.3f, PyTorch %.3f (diff %.2f)", i, logits[i], q.Logits[i], d)
					}
				}
			})
		}
	}
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
