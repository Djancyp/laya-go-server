package laya

import (
	"fmt"
	"os"
	"runtime"
	"strings"
	"testing"
)

func benchQuestions(n int) []Question {
	qs := make([]Question, n)
	for i := range qs {
		qs[i] = Question{
			ID: fmt.Sprintf("q%02d", i), Kind: Choice, Instructions: "Which topic does the text discuss?",
			Options: []Option{{"billing", "payments and invoices"}, {"tech", "bugs and outages"}, {"sales", "pricing and quotes"}, {"other", ""}},
		}
	}
	return qs
}

func BenchmarkBuildSequence(b *testing.B) {
	tok, head, cfg := benchSetup(b)
	m := &Model{tok: tok, head: head, cfg: *cfg}
	stateIDs := tok.Encode(strings.Repeat("Customer reports the invoice total is wrong. ", 40))
	q := benchQuestions(1)[0]
	b.ReportAllocs()
	for b.Loop() {
		if _, _, err := m.buildSequence(q, stateIDs); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecode(b *testing.B) {
	m := &Model{cfg: HeadConfig{Temperature: [3]float64{1, 1, 1}}}
	for _, k := range []int{2, 8, 64} {
		logits := make([]float32, k)
		for i := range logits {
			logits[i] = float32(i%5) - 2
		}
		q := Question{Kind: Choice, Options: make([]Option, k)}
		b.Run(fmt.Sprintf("options=%d", k), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				m.decode(q, logits)
			}
		})
	}
}

// BenchmarkPredict is the end-to-end cost (encoder + head) of one request across encoder
// contexts (questions encoded in parallel) and llama.cpp threads per context (0 = llama.cpp
// default). LAYA_BENCH_SWEEP=1 adds the full contexts x threads grid.
func BenchmarkPredict(b *testing.B) {
	benchSetup(b)
	lib := benchLlamaLib()
	if lib == "" {
		b.Skip("llama.cpp libs not found (set LAYA_LLAMA_LIB)")
	}
	if err := InitLlama(lib); err != nil {
		b.Skip(err)
	}
	state := strings.Repeat("Customer reports the invoice total is wrong. ", 20)
	type cfg struct{ ctxs, threads int }
	cfgs := []cfg{{1, 0}, {4, 0}}
	if os.Getenv("LAYA_BENCH_SWEEP") != "" {
		cpus := runtime.NumCPU()
		cfgs = nil
		for _, c := range []int{1, 2, 4} {
			seen := map[int]bool{}
			for _, t := range []int{0, 4, cpus / (2 * c), cpus / c, cpus} { // 0 = auto, 4 = llama.cpp's own default
				if t > 0 && !seen[t] || t == 0 {
					seen[t] = true
					cfgs = append(cfgs, cfg{c, t})
				}
			}
		}
	}
	for _, c := range cfgs {
		m, err := Open(Options{Dir: benchModel, Contexts: c.ctxs, Threads: c.threads})
		if err != nil {
			b.Fatal(err)
		}
		for _, nq := range []int{1, 4} {
			qs := benchQuestions(nq)
			b.Run(fmt.Sprintf("contexts=%d/threads=%d/questions=%d", c.ctxs, c.threads, nq), func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if _, err := m.Predict(state, qs); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
		m.Close()
	}
}
