package laya

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// The GLiNER2 tests need the model dir (LAYA_TEST_GLINER_DIR, default assets/gliner2-decide) and
// reference data from testdata/gen_ref_gliner.py; they skip without the model.
var glinerDir = envOr("LAYA_TEST_GLINER_DIR", "../assets/gliner2-decide")

type glinerRef struct {
	Text         string            `json:"text"`
	Task         string            `json:"task"`
	Labels       []string          `json:"labels"`
	Prompt       string            `json:"prompt"`
	Descriptions map[string]string `json:"descriptions"`
	IDs          []int32           `json:"ids"`
	Markers      []int             `json:"markers"`
	Logits       []float32         `json:"logits"`
}

func loadGlinerRef(t *testing.T) []glinerRef {
	t.Helper()
	raw, err := os.ReadFile("testdata/gliner/ref.json")
	if err != nil {
		t.Fatal(err)
	}
	var refs []glinerRef
	if err := json.Unmarshal(raw, &refs); err != nil {
		t.Fatal(err)
	}
	return refs
}

func glinerQuestion(c glinerRef) Question {
	q := Question{ID: c.Task, Kind: Choice, Instructions: c.Prompt}
	for _, l := range c.Labels {
		q.Options = append(q.Options, Option{Label: l, Description: c.Descriptions[l]})
	}
	return q
}

func testGlinerTokenizer(t *testing.T) *Gliner {
	t.Helper()
	if _, err := os.Stat(filepath.Join(glinerDir, "tokenizer.json")); err != nil {
		t.Skip(glinerDir, "/tokenizer.json not found")
	}
	tok, err := LoadUnigram(filepath.Join(glinerDir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	g := &Gliner{tok: tok, desc: "[DESCRIPTION]", maxTokens: 1 << 20}
	for name, dst := range map[string]*int32{"[P]": &g.p, "[L]": &g.l, "[SEP_TEXT]": &g.sep} {
		id, ok := tok.TokenID(name)
		if !ok {
			t.Fatalf("no %s token", name)
		}
		*dst = id
	}
	return g
}

func TestGlinerPromptMatchesHF(t *testing.T) {
	g := testGlinerTokenizer(t)
	for _, c := range loadGlinerRef(t) {
		q := glinerQuestion(c)
		labels, descs, err := glinerLabels(q)
		if err != nil {
			t.Fatal(err)
		}
		ids, markers := g.buildSequence(q, labels, descs, g.encodeText(c.Text))
		if !slices.Equal(ids, c.IDs) {
			t.Errorf("%q: ids differ\n got  %v\n want %v", c.Text, ids, c.IDs)
		}
		if !slices.Equal(markers, c.Markers) {
			t.Errorf("%q: markers %v, want %v", c.Text, markers, c.Markers)
		}
	}
}

func TestGlinerLogitsMatchHF(t *testing.T) {
	g := testGlinerTokenizer(t)
	gguf := filepath.Join(glinerDir, envOr("LAYA_GGUF", "gliner2-decide-encoder-F16.gguf"))
	if _, err := os.Stat(gguf); err != nil {
		t.Skip(gguf, "not found")
	}
	lib := os.Getenv("LAYA_LLAMA_LIB")
	if lib == "" {
		home, _ := os.UserHomeDir()
		lib = filepath.Join(home, ".kronk", "libraries", "linux", "amd64", "cpu")
	}
	if _, err := os.Stat(lib); err != nil {
		t.Skip("llama.cpp libraries not found in ", lib)
	}
	if err := InitLlama(lib); err != nil {
		t.Fatal(err)
	}
	m, err := OpenGliner(Options{Dir: glinerDir, GGUF: filepath.Base(gguf), Contexts: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	_ = g

	for _, c := range loadGlinerRef(t) {
		q := glinerQuestion(c)
		rs, err := m.Predict(c.Text, []Question{q})
		if err != nil {
			t.Fatal(err)
		}
		// Recompute the logits from the softmax the server returns: shift-invariant, so compare
		// log-probabilities against the reference logits' log-softmax.
		want := logSoftmax(c.Logits)
		for i, p := range rs[0].Probs {
			if d := math.Abs(math.Log(math.Max(p, 1e-12)) - want[i]); d > 0.05 {
				t.Errorf("%q label %d: log p %.4f, want %.4f", c.Text, i, math.Log(p), want[i])
			}
		}
		if got, want := rs[0].Choice, c.Labels[argmax(c.Logits)]; got != want {
			t.Errorf("%q: choice %q, want %q", c.Text, got, want)
		}
	}
}

func logSoftmax(x []float32) []float64 {
	m := math.Inf(-1)
	for _, v := range x {
		m = math.Max(m, float64(v))
	}
	var s float64
	for _, v := range x {
		s += math.Exp(float64(v) - m)
	}
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = float64(v) - m - math.Log(s)
	}
	return out
}

func argmax(x []float32) int {
	b := 0
	for i, v := range x {
		if v > x[b] {
			b = i
		}
	}
	return b
}
