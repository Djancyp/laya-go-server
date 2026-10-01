package laya

import (
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

// ErrInput marks a failure caused by the caller's question, not by the model or host.
var ErrInput = errors.New("laya: invalid question")

// Kind is the question type. The numeric values are the checkpoint's own type ids.
type Kind int

const (
	Choice Kind = 0 // pick one option
	Score  Kind = 1 // position on ordered levels
	Noul   Kind = 2 // yes/no probability
)

func (k Kind) String() string {
	switch k {
	case Choice:
		return "choice"
	case Score:
		return "score"
	case Noul:
		return "noul"
	}
	return "unknown"
}

// Option is one answer: a label plus an optional description that is shown to the model.
type Option struct {
	Label       string
	Description string
}

// Question is one typed judgment about the state.
//
//	Choice: Options are the candidate labels (order matters slightly; keep it stable).
//	Score:  Options are the ordered levels; only Description is used ("level i: ...").
//	Noul:   Options may be empty; two entries (false, true) supply their descriptions.
type Question struct {
	ID           string
	Kind         Kind
	Instructions string
	Options      []Option
}

// Result is the answer to one question.
type Result struct {
	ID    string
	Kind  Kind
	Probs []float64 // probability per option in Question.Options order (Noul: [false, true])

	Choice string  // Choice: winning label
	Score  float64 // Score: expected level (probability-weighted index)
	Noul   float64 // Noul: probability of yes

	// Confidence follows the Jev convention: how concentrated the distribution is
	// (1 - normalized entropy) for Choice/Score, max(p, 1-p) for Noul.
	Confidence float64
	// AnswerConfidence is max(p), the quantity the checkpoint's temperature is calibrated on.
	AnswerConfidence float64

	Tokens int // encoder input length for this question
}

// Model bundles tokenizer, encoder and head.
type Model struct {
	tok  *Tokenizer
	enc  *Encoder
	head *Head
	cfg  HeadConfig
}

// Options configures Open.
type Options struct {
	Dir       string // directory with tokenizer.json and the head safetensors
	GGUF      string // encoder GGUF file name inside Dir (default laya-multilingual-F16.gguf)
	GPULayers int    // encoder layers offloaded to the GPU (0 = CPU)
	Threads   int    // CPU threads per encoder context (0 = GOMAXPROCS / Contexts)
	Contexts  int    // encoder contexts, i.e. questions encoded in parallel (0 = 1)
}

// Open loads the tokenizer, head and encoder. llama.cpp must be initialised first
// (see InitLlama).
func Open(o Options) (*Model, error) {
	if o.GGUF == "" {
		o.GGUF = "laya-multilingual-F16.gguf"
	}
	if o.Threads <= 0 {
		// llama.cpp defaults to 4 threads per context, leaving most cores idle. Splitting
		// GOMAXPROCS (cgroup-aware) across contexts cut a 1-context request 327 -> 220 ms
		// on 16 threads; 8 vs 16 threads measured the same (memory-bound), so SMT is harmless.
		o.Threads = max(1, runtime.GOMAXPROCS(0)/max(1, o.Contexts))
	}

	tok, err := LoadTokenizer(filepath.Join(o.Dir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	head, cfg, err := LoadHead(filepath.Join(o.Dir, "laya-multilingual-head.safetensors"))
	if err != nil {
		return nil, err
	}
	enc, err := NewEncoder(filepath.Join(o.Dir, o.GGUF), cfg.MaxLen, o.GPULayers, o.Contexts, o.Threads)
	if err != nil {
		return nil, err
	}
	if enc.Dim() != head.Dim() {
		enc.Close()
		return nil, fmt.Errorf("laya: encoder %s has hidden size %d but the head expects %d: the GGUF and head files are from different models",
			o.GGUF, enc.Dim(), head.Dim())
	}
	return &Model{tok: tok, enc: enc, head: head, cfg: *cfg}, nil
}

// Close frees the encoder.
func (m *Model) Close() { m.enc.Close() }

// Predict answers every question about state. state is plain text (serialize structured
// data yourself: the model sees exactly this string).
func (m *Model) Predict(state string, questions []Question) ([]Result, error) {
	if len(questions) == 0 {
		return nil, errors.New("laya: no questions")
	}

	stateIDs := m.tok.Encode(strings.ReplaceAll(state, m.tok.MaskText, " "))
	results := make([]Result, len(questions))
	errs := make([]error, len(questions))

	// One goroutine per question, bounded by the encoder's idle contexts: Hidden blocks
	// for a free one, and the tokenized state is shared read-only.
	var wg sync.WaitGroup
	sem := make(chan struct{}, m.enc.Contexts())
	for i, q := range questions {
		sem <- struct{}{}
		wg.Add(1)
		go func() {
			defer func() {
				if v := recover(); v != nil {
					errs[i] = fmt.Errorf("laya: question %q: panic: %v", q.ID, v)
				}
				<-sem
				wg.Done()
			}()
			results[i], errs[i] = m.predictOne(q, stateIDs)
		}()
	}
	wg.Wait()
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	return results, nil
}

func (m *Model) predictOne(q Question, stateIDs []int32) (Result, error) {
	ids, markers, err := m.buildSequence(q, stateIDs)
	if err != nil {
		return Result{}, err
	}
	hidden, err := m.enc.Hidden(ids)
	if err != nil {
		return Result{}, fmt.Errorf("laya: question %q: %w", q.ID, err)
	}
	logits, err := m.head.Logits(hidden, int(q.Kind), markers)
	if err != nil {
		return Result{}, fmt.Errorf("laya: question %q: %w", q.ID, err)
	}
	want := len(q.Options)
	if q.Kind == Noul {
		want = 2
	}
	if len(logits) != want {
		return Result{}, fmt.Errorf("laya: question %q: head returned %d logits, want %d", q.ID, len(logits), want)
	}
	r := m.decode(q, logits)
	r.Tokens = len(ids)
	return r, nil
}

// renderOptions is the text shown for each option (Python laya render_options).
func renderOptions(q Question) ([]string, error) {
	switch q.Kind {
	case Choice:
		if len(q.Options) < 1 {
			return nil, fmt.Errorf("laya: question %q needs at least one option", q.ID)
		}
		out := make([]string, len(q.Options))
		for i, o := range q.Options {
			out[i] = o.Label
			if o.Description != "" {
				out[i] = o.Label + ": " + o.Description
			}
		}
		return out, nil
	case Score:
		if len(q.Options) < 2 {
			return nil, fmt.Errorf("laya: score question %q needs at least two levels", q.ID)
		}
		out := make([]string, len(q.Options))
		for i, o := range q.Options {
			out[i] = fmt.Sprintf("level %d: %s", i, o.Description)
		}
		return out, nil
	case Noul:
		no, yes := "no, the statement does not hold", "yes, the statement holds"
		if len(q.Options) == 2 {
			if d := q.Options[0].Description; d != "" {
				no = d
			}
			if d := q.Options[1].Description; d != "" {
				yes = d
			}
		}
		return []string{"false: " + no, "true: " + yes}, nil
	}
	return nil, fmt.Errorf("laya: question %q has unknown kind %d", q.ID, q.Kind)
}

// buildSequence lays out: [CLS] "<type> question: <ins>" [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
// with the checkpoint's budgets (a port of laya.common.build_sequence).
func (m *Model) buildSequence(q Question, stateIDs []int32) (ids []int32, markers []int, err error) {
	opts, err := renderOptions(q)
	if err != nil {
		return nil, nil, err
	}
	maxLen, headMax := m.cfg.MaxLen, m.cfg.HeadMaxLen

	ins := strings.ReplaceAll(q.Instructions, m.tok.MaskText, " ")
	headIDs := m.tok.Encode(fmt.Sprintf("%s question: %s", q.Kind, ins))

	optIDs := make([][]int32, len(opts))
	total := 0
	for i, o := range opts {
		toks := m.tok.Encode(" " + strings.ReplaceAll(o, m.tok.MaskText, " "))
		toks = toks[:min(len(toks), 48)]
		optIDs[i] = append([]int32{m.tok.Mask}, toks...)
		total += len(optIDs[i])
	}

	budget := headMax - total
	if budget < 16 {
		per := max(4, (headMax-16)/max(1, len(optIDs)))
		total = 0
		for i, o := range optIDs {
			optIDs[i] = o[:min(len(o), per)]
			total += len(optIDs[i])
		}
		budget = headMax - total
	}
	headIDs = headIDs[:min(len(headIDs), max(8, budget))]

	ids = append(ids, m.tok.CLS)
	ids = append(ids, headIDs...)
	ids = append(ids, m.tok.SEP)
	for _, o := range optIDs {
		markers = append(markers, len(ids))
		ids = append(ids, o...)
	}
	ids = append(ids, m.tok.SEP)

	room := max(0, maxLen-len(ids)-1)
	ids = append(ids, stateIDs[:min(len(stateIDs), room)]...)
	ids = append(ids, m.tok.SEP)

	if len(ids) > maxLen {
		ids = ids[:maxLen]
	}
	kept := markers[:0]
	for _, mk := range markers {
		if mk < maxLen {
			kept = append(kept, mk)
		}
	}
	if len(kept) != len(opts) {
		return nil, nil, fmt.Errorf("%w: question %q: options exceed head_max_len=%d", ErrInput, q.ID, headMax)
	}
	return ids, kept, nil
}

// decode turns logits into a Result (temperature scaling, softmax, typed answer).
func (m *Model) decode(q Question, logits []float32) Result {
	k := len(logits)
	t := m.temperature(q.Kind, k)

	p := make([]float64, k)
	maxZ := math.Inf(-1)
	for _, l := range logits {
		maxZ = math.Max(maxZ, float64(l)/t)
	}
	var sum float64
	for i, l := range logits {
		p[i] = math.Exp(float64(l)/t - maxZ)
		sum += p[i]
	}
	best := 0
	for i := range p {
		p[i] /= sum
		if p[i] > p[best] {
			best = i
		}
	}

	r := Result{ID: q.ID, Kind: q.Kind, Probs: p, AnswerConfidence: p[best]}
	switch q.Kind {
	case Choice:
		r.Choice = q.Options[best].Label
		r.Confidence = entropyConfidence(p)
	case Score:
		for i, v := range p {
			r.Score += float64(i) * v
		}
		r.Confidence = entropyConfidence(p)
	case Noul:
		r.Noul = p[1]
		r.Confidence = math.Max(p[1], 1-p[1])
	}
	return r
}

// entropyConfidence is 1 - H(p)/log(k): 1 when one option has all the mass, 0 when uniform.
func entropyConfidence(p []float64) float64 {
	if len(p) < 2 {
		return 1
	}
	var h float64
	for _, v := range p {
		h -= v * math.Log(math.Max(v, 1e-12))
	}
	return math.Min(1, math.Max(0, 1-h/math.Log(float64(len(p)))))
}

// temperature is the checkpoint's calibration temperature for this question shape,
// clamped to [0.5, 5] like the reference implementation.
func (m *Model) temperature(kind Kind, k int) float64 {
	size := "11+"
	switch {
	case k <= 2:
		size = "2"
	case k <= 5:
		size = "3-5"
	case k <= 10:
		size = "6-10"
	}
	t := m.cfg.Temperature[kind]
	if v, ok := m.cfg.TemperatureByOptions[kind.String()+":"+size]; ok {
		t = v
	}
	if math.IsNaN(t) || math.IsInf(t, 0) {
		return 1
	}
	return math.Min(5, math.Max(0.5, t))
}
