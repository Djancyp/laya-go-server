package laya

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
)

// Predictor is what a loaded model offers: Model (Laya) and Gliner (GLiNER2.5-Decide).
type Predictor interface {
	Predict(state string, questions []Question) ([]Result, error)
	Close()
}

// glinerHead is the name of the head file that marks a model dir as a GLiNER2 one.
const glinerHead = "gliner2-decide-head.safetensors"

// glinerMaxTokens bounds one encoder sequence when Options.MaxTokens is unset.
const glinerMaxTokens = 1024

// OpenAny opens the model in o.Dir, a GLiNER2 one when it holds gliner2-decide-head.safetensors,
// a Qwen3Guard one when it holds a *Qwen3Guard*.gguf, else a Laya one. llama.cpp must be initialised first (see InitLlama).
func OpenAny(o Options) (Predictor, error) {
	if _, err := os.Stat(filepath.Join(o.Dir, glinerHead)); err == nil {
		return OpenGliner(o)
	}
	if m, _ := filepath.Glob(filepath.Join(o.Dir, qwenGuardGlob)); len(m) > 0 {
		return OpenQwenGuard(o)
	}
	return Open(o)
}

// Gliner is GLiNER2.5-Decide: a DeBERTa-v3-large encoder (GGUF via llama.cpp) whose [L] label
// markers are scored by a two-layer MLP. Every question becomes one gliner2 classification task:
//
//	([P] <id>: <instructions> [DESCRIPTION] <label>: <desc> ... ([L] <label> [L] <label> ... ) )[SEP_TEXT] <state words>
type Gliner struct {
	tok       *Unigram
	enc       *Encoder
	head      mlpHead
	p, l, sep int32 // ids of [P], [L], [SEP_TEXT]
	desc      string
	maxTokens int
}

// mlpHead is classifier.0 (relu) -> classifier.2: one logit per [L] marker.
type mlpHead struct {
	d, h   int
	w1, b1 []float32 // h x d, h
	w2     []float32 // h
	b2     float32
}

func loadMLPHead(path string) (mlpHead, error) {
	t, err := LoadSafetensors(path)
	if err != nil {
		return mlpHead{}, err
	}
	shape, ok := t.Shape("classifier.0.weight")
	if !ok || len(shape) != 2 {
		return mlpHead{}, fmt.Errorf("laya: classifier.0.weight has shape %v, want [h d]", shape)
	}
	h := mlpHead{h: shape[0], d: shape[1]}
	var firstErr error
	get := func(name string, shape ...int) []float32 {
		v, err := t.Get(name, shape...)
		if err != nil && firstErr == nil {
			firstErr = err
		}
		return v
	}
	h.w1, h.b1 = get("classifier.0.weight", h.h, h.d), get("classifier.0.bias", h.h)
	h.w2 = get("classifier.2.weight", 1, h.h)
	b2 := get("classifier.2.bias", 1)
	if firstErr != nil {
		return mlpHead{}, firstErr
	}
	h.b2 = b2[0]
	return h, nil
}

// logits scores the token at each marker position of hidden (n x d, row-major).
func (h *mlpHead) logits(hidden []float32, markers []int) ([]float32, error) {
	if len(hidden) == 0 || len(hidden)%h.d != 0 {
		return nil, fmt.Errorf("laya: hidden has %d values, not a multiple of %d", len(hidden), h.d)
	}
	n := len(hidden) / h.d
	out := make([]float32, len(markers))
	act := make([]float32, h.h)
	for i, m := range markers {
		if m < 0 || m >= n {
			return nil, fmt.Errorf("laya: marker %d outside sequence of %d tokens", m, n)
		}
		x := hidden[m*h.d : (m+1)*h.d]
		for j := range act {
			act[j] = max(0, dot(x, h.w1[j*h.d:(j+1)*h.d])+h.b1[j])
		}
		out[i] = dot(act, h.w2) + h.b2
	}
	return out, nil
}

// OpenGliner loads tokenizer, head and encoder from o.Dir. o.GGUF must name the encoder GGUF.
func OpenGliner(o Options) (*Gliner, error) {
	if o.GGUF == "" {
		return nil, errors.New("laya: gliner model needs the encoder GGUF name")
	}
	if o.MaxTokens <= 0 {
		o.MaxTokens = glinerMaxTokens
	}
	if o.Threads <= 0 {
		o.Threads = max(1, runtime.GOMAXPROCS(0)/max(1, o.Contexts))
	}
	tok, err := LoadUnigram(filepath.Join(o.Dir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	g := &Gliner{tok: tok, desc: "[DESCRIPTION]", maxTokens: o.MaxTokens}
	for name, dst := range map[string]*int32{"[P]": &g.p, "[L]": &g.l, "[SEP_TEXT]": &g.sep} {
		id, ok := tok.TokenID(name)
		if !ok {
			return nil, fmt.Errorf("laya: tokenizer has no %s token", name)
		}
		*dst = id
	}
	if g.head, err = loadMLPHead(filepath.Join(o.Dir, glinerHead)); err != nil {
		return nil, err
	}
	if g.enc, err = NewEncoder(filepath.Join(o.Dir, o.GGUF), o.MaxTokens, o.GPULayers, o.Contexts, o.Threads); err != nil {
		return nil, err
	}
	if g.enc.Dim() != g.head.d {
		g.enc.Close()
		return nil, fmt.Errorf("laya: encoder %s has hidden size %d but the head expects %d: the GGUF and head files are from different models",
			o.GGUF, g.enc.Dim(), g.head.d)
	}
	return g, nil
}

// Close frees the encoder.
func (g *Gliner) Close() { g.enc.Close() }

// Predict answers every question about state (plain text, as for Model.Predict).
func (g *Gliner) Predict(state string, questions []Question) ([]Result, error) {
	if len(questions) == 0 {
		return nil, errors.New("laya: no questions")
	}
	stateIDs := g.encodeText(state)
	results := make([]Result, len(questions))
	errs := make([]error, len(questions))

	var wg sync.WaitGroup
	sem := make(chan struct{}, g.enc.Contexts())
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
			results[i], errs[i] = g.predictOne(q, stateIDs)
		}()
	}
	wg.Wait()
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	return results, nil
}

func (g *Gliner) predictOne(q Question, stateIDs []int32) (Result, error) {
	labels, descs, err := glinerLabels(q)
	if err != nil {
		return Result{}, err
	}
	ids, markers := g.buildSequence(q, labels, descs, stateIDs)
	if len(markers) != len(labels) {
		return Result{}, fmt.Errorf("%w: question %q: labels exceed the %d token limit", ErrInput, q.ID, g.maxTokens)
	}
	hidden, err := g.enc.Hidden(ids)
	if err != nil {
		return Result{}, fmt.Errorf("laya: question %q: %w", q.ID, err)
	}
	logits, err := g.head.logits(hidden, markers)
	if err != nil {
		return Result{}, fmt.Errorf("laya: question %q: %w", q.ID, err)
	}
	r := decodeLogits(q, logits, 1) // single-label softmax, no calibration
	r.Tokens = len(ids)
	return r, nil
}

// glinerLabels maps a question to gliner2 labels and their descriptions (parallel slices).
func glinerLabels(q Question) (labels, descs []string, err error) {
	switch q.Kind {
	case Choice:
		if len(q.Options) < 1 {
			return nil, nil, fmt.Errorf("laya: question %q needs at least one option", q.ID)
		}
		for _, o := range q.Options {
			labels, descs = append(labels, o.Label), append(descs, o.Description)
		}
	case Score:
		if len(q.Options) < 2 {
			return nil, nil, fmt.Errorf("laya: score question %q needs at least two levels", q.ID)
		}
		for i, o := range q.Options {
			labels, descs = append(labels, fmt.Sprintf("level %d", i)), append(descs, o.Description)
		}
	case Noul:
		labels, descs = []string{"false", "true"}, []string{"", ""}
		if len(q.Options) == 2 {
			descs = []string{q.Options[0].Description, q.Options[1].Description}
		}
	default:
		return nil, nil, fmt.Errorf("laya: question %q has unknown kind %d", q.ID, q.Kind)
	}
	return labels, descs, nil
}

// buildSequence lays out the gliner2 prompt (a port of its SchemaTransformer for one
// classification task). It returns the ids and the position of each label's [L] marker;
// the state is cut to fit maxTokens.
func (g *Gliner) buildSequence(q Question, labels, descs []string, stateIDs []int32) (ids []int32, markers []int) {
	task := g.scrub(q.ID)
	if ins := g.scrub(q.Instructions); ins != "" {
		task += ": " + ins
	}
	for i, l := range labels {
		if descs[i] != "" {
			task += " " + g.desc + " " + g.scrub(l) + ": " + g.scrub(descs[i])
		}
	}
	punct := func(s string) []int32 { return g.tok.Encode(s) }

	ids = append(ids, punct("(")...)
	ids = append(ids, g.p)
	ids = append(ids, g.tok.Encode(task)...)
	ids = append(ids, punct("(")...)
	for _, l := range labels {
		markers = append(markers, len(ids))
		ids = append(ids, g.l)
		ids = append(ids, g.tok.Encode(g.scrub(l))...)
	}
	ids = append(ids, punct(")")...)
	ids = append(ids, punct(")")...)
	ids = append(ids, g.sep)

	if len(ids) >= g.maxTokens {
		kept := markers[:0]
		for _, m := range markers {
			if m < g.maxTokens {
				kept = append(kept, m)
			}
		}
		return ids[:g.maxTokens], kept
	}
	ids = append(ids, stateIDs[:min(len(stateIDs), g.maxTokens-len(ids))]...)
	return ids, markers
}

// scrub removes added-token text ("[L]", "[P]", ...) that would otherwise be read as markers.
func (g *Gliner) scrub(s string) string {
	for _, a := range g.tok.added {
		s = strings.ReplaceAll(s, a.content, " ")
	}
	return s
}

// wordPattern is gliner2's WhitespaceTokenSplitter regex; Go's \w is ASCII only, so it is spelled out.
var wordPattern = regexp.MustCompile(`(?i)(?:https?://\S+|www\.\S+)` +
	`|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}` +
	`|@[a-z0-9_]+` +
	`|[\p{L}\p{N}_]+(?:[-_][\p{L}\p{N}_]+)*` +
	`|\S`)

// encodeText is gliner2's text handling: end with punctuation, split into lowercased words,
// tokenize each word on its own.
func (g *Gliner) encodeText(text string) []int32 {
	if text == "" || !strings.HasSuffix(text, ".") && !strings.HasSuffix(text, "!") && !strings.HasSuffix(text, "?") {
		text += "."
	}
	var ids []int32
	for _, w := range wordPattern.FindAllString(g.scrub(text), -1) {
		ids = append(ids, g.tok.Encode(pyLower(w))...)
		if len(ids) > g.maxTokens {
			break // the rest cannot fit
		}
	}
	return ids
}

// pyLower is Python's str.lower(), which differs from Go's only for U+0130 (-> i + combining dot).
func pyLower(s string) string {
	return strings.ToLower(strings.ReplaceAll(s, "İ", "i̇"))
}
