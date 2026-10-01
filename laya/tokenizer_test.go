package laya

import (
	"encoding/json"
	"os"
	"slices"
	"sync"
	"testing"
)

// Large files, downloaded separately; tests skip without them. LAYA_TEST_MODEL_DIR / LAYA_TEST_REF_DIR
// (and LAYA_GGUF) point the whole suite at another checkpoint, e.g. the English ModernBERT one:
//
//	LAYA_TEST_MODEL_DIR=../models/laya-en LAYA_TEST_REF_DIR=testdata/en LAYA_GGUF=laya-en-F16.gguf go test ./laya
var (
	modelDir = envOr("LAYA_TEST_MODEL_DIR", "../models/laya")
	refDir   = envOr("LAYA_TEST_REF_DIR", "testdata")
)

var (
	tokOnce sync.Once
	tokInst *Tokenizer
	tokErr  error
)

// testTokenizer loads the 34 MB tokenizer once for the whole test binary.
func testTokenizer(t *testing.T) *Tokenizer {
	t.Helper()
	if _, err := os.Stat(modelDir + "/tokenizer.json"); err != nil {
		t.Skip(modelDir, "/tokenizer.json not downloaded")
	}
	tokOnce.Do(func() { tokInst, tokErr = LoadTokenizer(modelDir + "/tokenizer.json") })
	if tokErr != nil {
		t.Fatal(tokErr)
	}
	return tokInst
}

func TestTokenizerMatchesHF(t *testing.T) {
	tok := testTokenizer(t)

	raw, err := os.ReadFile(refDir + "/tokenizer_ref.json")
	if err != nil {
		t.Skip("run laya/testdata/gen_ref.py first")
	}
	var cases []struct {
		Text string  `json:"text"`
		IDs  []int32 `json:"ids"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}

	for _, c := range cases {
		name := c.Text
		if len(name) > 30 {
			name = name[:30]
		}
		t.Run(name, func(t *testing.T) {
			if got := tok.Encode(c.Text); !slices.Equal(got, c.IDs) {
				t.Fatalf("Encode(%q)\n got %v\nwant %v", c.Text, got, c.IDs)
			}
		})
	}
}
