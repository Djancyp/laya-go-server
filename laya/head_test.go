package laya

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"slices"
	"sync"
	"testing"
)

// refCase mirrors one entry of testdata/predict_ref.json (written by testdata/gen_ref.py).
type refCase struct {
	State        json.RawMessage `json:"state"`
	Questions    json.RawMessage `json:"questions"`
	PerQuestion  map[string]refQuestion
	Answers      map[string]map[string]any `json:"answers"`
	QuestionsRaw map[string]json.RawMessage
}

type refQuestion struct {
	IDs     []int32   `json:"ids"`
	Markers []int     `json:"markers"`
	QType   int       `json:"qtype"`
	Hidden  string    `json:"hidden"` // float32 LE, base64, n x d
	N       int       `json:"n"`
	D       int       `json:"d"`
	Logits  []float64 `json:"logits"`
}

func (r refQuestion) hidden(t *testing.T) []float32 {
	t.Helper()
	b, err := base64.StdEncoding.DecodeString(r.Hidden)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	if len(out) != r.N*r.D {
		t.Fatalf("hidden has %d values, want %d", len(out), r.N*r.D)
	}
	return out
}

var (
	refOnce  sync.Once
	refCases []struct {
		State       json.RawMessage            `json:"state"`
		Questions   map[string]json.RawMessage `json:"questions"`
		PerQuestion map[string]refQuestion     `json:"per_question"`
		Answers     map[string]map[string]any  `json:"answers"`
	}
	refErr error
)

func loadRef(t *testing.T) {
	t.Helper()
	refOnce.Do(func() {
		raw, err := os.ReadFile("testdata/predict_ref.json")
		if err != nil {
			refErr = err
			return
		}
		refErr = json.Unmarshal(raw, &refCases)
	})
	if refErr != nil {
		t.Skip("run laya/testdata/gen_ref.py first: ", refErr)
	}
}

func testHead(t *testing.T) (*Head, *HeadConfig) {
	t.Helper()
	if _, err := os.Stat(modelDir + "/laya-multilingual-head.safetensors"); err != nil {
		t.Skip("models/laya head not downloaded")
	}
	h, cfg, err := LoadHead(modelDir + "/laya-multilingual-head.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	return h, cfg
}

func TestHeadMatchesPyTorch(t *testing.T) {
	loadRef(t)
	h, _ := testHead(t)

	for ci, c := range refCases {
		for qid, q := range c.PerQuestion {
			t.Run(qid+"/case"+string(rune('a'+ci)), func(t *testing.T) {
				got, err := h.Logits(q.hidden(t), q.QType, q.Markers)
				if err != nil {
					t.Fatal(err)
				}
				if len(got) != len(q.Logits) {
					t.Fatalf("%d logits, want %d", len(got), len(q.Logits))
				}
				for i := range got {
					if diff := math.Abs(float64(got[i]) - q.Logits[i]); diff > 2e-3 {
						t.Errorf("logit %d = %.5f, want %.5f (diff %.2g)", i, got[i], q.Logits[i], diff)
					}
				}
			})
		}
	}
}

func TestHeadRejectsBadInput(t *testing.T) {
	h, _ := testHead(t)
	good := make([]float32, 4*hiddenDim)

	tests := []struct {
		name    string
		hidden  []float32
		qtype   int
		markers []int
	}{
		{"empty hidden", nil, 0, []int{0}},
		{"ragged hidden", make([]float32, hiddenDim+1), 0, []int{0}},
		{"bad qtype", good, 3, []int{0}},
		{"negative qtype", good, -1, []int{0}},
		{"marker past the end", good, 0, []int{4}},
		{"negative marker", good, 0, []int{-1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := h.Logits(tt.hidden, tt.qtype, tt.markers); err == nil {
				t.Fatal("want an error")
			}
		})
	}
}

// TestLinearMatchesReference checks the tiled, parallel linear against a plain per-row
// reference, including a partial output tile.
func TestLinearMatchesReference(t *testing.T) {
	const in, out = 37, 70 // odd sizes exercise the dot tail and a partial tile
	w, b := make([]float32, out*in), make([]float32, out)
	for i := range w {
		w[i] = float32(i%13)/13 - 0.5
	}
	for i := range b {
		b[i] = float32(i) / 10
	}
	for _, rows := range []int{1, 3, 4, 5, 8, 11} {
		x := make([]float32, rows*in)
		for i := range x {
			x[i] = float32(i%7)/7 - 0.3
		}
		got := linear(nil, x, rows, in, w, b, out)
		for r := range rows {
			for o := range out {
				want := dot(x[r*in:(r+1)*in], w[o*in:(o+1)*in]) + b[o]
				if d := math.Abs(float64(got[r*out+o] - want)); d > 1e-5 {
					t.Fatalf("rows=%d y[%d][%d] = %v, want %v", rows, r, o, got[r*out+o], want)
				}
			}
		}
	}
}

// TestLogitsScratchReuse runs Logits on inputs of different lengths so pooled scratch is
// reused, grown and shrunk, and checks repeated inputs give bit-identical logits: stale
// arena contents or a pooled buffer escaping into a result would break this.
func TestLogitsScratchReuse(t *testing.T) {
	h, _ := testHead(t)
	in := func(n int, seed float32) []float32 {
		x := make([]float32, n*hiddenDim)
		for i := range x {
			x[i] = float32(math.Sin(float64(seed) + float64(i)*0.001))
		}
		return x
	}
	markers := []int{1, 3, 5}
	cases := []struct {
		n    int
		seed float32
	}{{40, 1}, {300, 2}, {12, 3}, {300, 4}}
	first := make([][]float32, len(cases))
	for round := range 2 {
		for i, c := range cases {
			got, err := h.Logits(in(c.n, c.seed), int(Choice), markers)
			if err != nil {
				t.Fatal(err)
			}
			if round == 0 {
				first[i] = got
				continue
			}
			if !slices.Equal(got, first[i]) {
				t.Errorf("case %d: second run %v, first %v", i, got, first[i])
			}
		}
	}
}
