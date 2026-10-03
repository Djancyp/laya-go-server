package laya

import (
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/hybridgroup/yzma/pkg/llama"
)

// qwenGuardGlob marks a model dir as a Qwen3Guard-Gen one.
const qwenGuardGlob = "*Qwen3Guard*.gguf"

// qwenGuardMaxTokens bounds one prompt when Options.MaxTokens is unset.
const qwenGuardMaxTokens = 4096

// Qwen3Guard-Gen's chat template (user-query branch), split around the conversation. The model
// would answer "Safety: Safe|Unsafe|Controversial\nCategories: ...": the prompt ends after
// "Safety:" and the next-token logits of the three verdicts are the whole answer, so no
// generation loop runs.
const (
	qwenGuardHead = "<|im_start|>user\n# Task:\n" +
		"1) Evaluate whether THE LAST USER's query belongs to safe or unsafe content based on our safety policy.\n" +
		"2) Identify the category of unsafe content found.\n\n" +
		"<BEGIN SAFETY POLICY>\n" +
		"**Safety: Safe**: The USER's query does not contain any prohibited content or content that could lead to prohibited content.\n" +
		"**Safety: Unsafe**: The USER's query contains prohibited content.\n" +
		"**Safety: Controversial**: The USER's query does not explicitly contain prohibited content, but its intent, context, or potential responses could be misinterpreted, misused, or exploited to generate unsafe content under certain conditions.\n" +
		"<END SAFETY POLICY>\n\n" +
		"<BEGIN UNSAFE CONTENT CATEGORIES>\n" +
		"Violent.\nNon-violent Illegal Acts.\nSexual Content or Sexual Acts.\nPII.\nSuicide & Self-Harm.\nUnethical Acts.\nPolitically Sensitive Topics.\nCopyright Violation.\nJailbreak.\n" +
		"<END UNSAFE CONTENT CATEGORIES>\n\n" +
		"<BEGIN CONVERSATION>\n\n" +
		"USER: "
	qwenGuardTail = "\n\n<END CONVERSATION>\n\n" +
		"Provide your safety assessment for ONLY THE LAST **USER's query** in the above conversation:\n" +
		" - The first line must be one of: 'Safety: Safe', 'Safety: Unsafe', 'Safety: Controversial'.\n" +
		" - The second line should start with 'Categories:' followed by a list of any unsafe content categories, separated by commas. If the content is safe, use 'Categories: None'.<|im_end|>\n" +
		"<|im_start|>assistant\n<think>\n\n</think>\n\n" +
		"Safety:"
)

// The response-moderation branch of the template: it also asks for the refusal line.
const (
	qwenGuardRespHead = "<|im_start|>user\n# Task:\n" +
		"1) Evaluate whether THE LAST ASSISTANT's response belongs to safe or unsafe content based on our safety policy.\n" +
		"2) Identify the category of unsafe content found.\n" +
		"3) Determine if THE LAST ASSISTANT's response is a refusal to the USER's query.\n\n" +
		"<BEGIN SAFETY POLICY>\n" +
		"**Safety: Safe**: The ASSISTANT's response does not contain any prohibited content or content that could lead to prohibited content.\n" +
		"**Safety: Unsafe**: The ASSISTANT's response contains prohibited content.\n" +
		"**Safety: Controversial**: The ASSISTANT's response is not explicitly harmful, but could become risky or offensive depending on context or audience.\n" +
		"<END SAFETY POLICY>\n\n" +
		"<BEGIN UNSAFE CONTENT CATEGORIES>\n" +
		"Violent.\nNon-violent Illegal Acts.\nSexual Content or Sexual Acts.\nPII.\nSuicide & Self-Harm.\nUnethical Acts.\nPolitically Sensitive Topics.\nCopyright Violation.\n" +
		"<END UNSAFE CONTENT CATEGORIES>\n\n" +
		"<BEGIN REFUSAL POLICY>\n" +
		"**Refusal: Yes**: The ASSISTANT refused to answer or did not comply with the USER's request.\n" +
		"**Refusal: No**: The ASSISTANT directly answered or complied with the USER's request.\n" +
		"<END REFUSAL POLICY>\n\n" +
		"<BEGIN CONVERSATION>\n\n" +
		"USER: "
	qwenGuardRespMid  = "\n\nASSISTANT: "
	qwenGuardRespTail = "\n\n<END CONVERSATION>\n\n" +
		"Provide your safety assessment for ONLY THE LAST **ASSISTANT's response** in the above conversation:\n" +
		" - The first line must be one of: 'Safety: Safe', 'Safety: Unsafe', 'Safety: Controversial'.\n" +
		" - The second line should start with 'Categories:' followed by a list of any unsafe content categories, separated by commas. If the content is safe, use 'Categories: None'.\n" +
		" - The third line must be one of: 'Refusal: Yes', 'Refusal: No'.<|im_end|>\n" +
		"<|im_start|>assistant\n<think>\n\n</think>\n\n" +
		"Safety:"
)

// qwenGuardCategories are the unsafe-content categories the model names.
var qwenGuardCategories = []string{
	"Violent", "Non-violent Illegal Acts", "Sexual Content or Sexual Acts", "PII", "Suicide & Self-Harm",
	"Unethical Acts", "Politically Sensitive Topics", "Copyright Violation", "Jailbreak",
}

// qwenGuardMaxGen bounds the generated assessment (three short lines).
const qwenGuardMaxGen = 48

// Verdicts, in the order of QwenGuard.verdict.
var qwenVerdicts = [3]string{"Safe", "Unsafe", "Controversial"}

// guardCtx is a llama context plus which template head its KV cache holds (0 none, 1 prompt, 2 response):
// the head is the same for every request, so only the text after it is decoded.
type guardCtx struct {
	ctx    llama.Context
	cached int
}

// QwenGuard is Qwen3Guard-Gen (a causal LM, GGUF via llama.cpp) used as a prompt-safety
// classifier: the state is the user query, and the verdict token probabilities answer questions:
//
//	noul:   P(yes) = P(unsafe) + P(controversial)
//	choice: options labelled safe / unsafe / controversial (any subset, case-insensitive)
//
// The question's instructions and descriptions are ignored: the policy is the model's own.
type QwenGuard struct {
	model     llama.Model
	vocab     llama.Vocab
	pool      chan *guardCtx
	n, made   int
	nVocab    int
	maxTokens int
	head      []llama.Token
	tail      []llama.Token
	respHead  []llama.Token
	respMid   []llama.Token
	respTail  []llama.Token
	verdict   [3]llama.Token // first token of " Safe", " Unsafe", " Controversial"
}

// OpenQwenGuard loads the Qwen3Guard GGUF in o.Dir. llama.cpp must be initialised first.
func OpenQwenGuard(o Options) (*QwenGuard, error) {
	path := filepath.Join(o.Dir, o.GGUF)
	if o.GGUF == "" || o.GGUF == "laya-multilingual-F16.gguf" {
		m, _ := filepath.Glob(filepath.Join(o.Dir, qwenGuardGlob))
		if len(m) != 1 {
			return nil, fmt.Errorf("laya: want exactly one %s in %s, found %d", qwenGuardGlob, o.Dir, len(m))
		}
		path = m[0]
	}
	contexts := max(1, o.Contexts)
	if o.Threads <= 0 {
		// Physical cores only: decode is memory-bound and prefill compute-bound, so SMT siblings
		// only add contention (Ryzen 7 8845HS, 0.6B Q8_0: 8 threads beat 16 by ~25%, 12 was worst).
		o.Threads = max(1, runtime.GOMAXPROCS(0)/2/contexts)
	}
	if o.MaxTokens <= 0 {
		o.MaxTokens = qwenGuardMaxTokens
	}

	mp := llama.ModelDefaultParams()
	mp.NGpuLayers = int32(o.GPULayers)
	model, err := llama.ModelLoadFromFile(path, mp)
	if err != nil {
		return nil, fmt.Errorf("laya: load %s: %w", path, err)
	}
	g := &QwenGuard{model: model, vocab: llama.ModelGetVocab(model), pool: make(chan *guardCtx, contexts), n: contexts, maxTokens: o.MaxTokens}
	g.nVocab = int(llama.VocabNTokens(g.vocab))
	g.head = llama.Tokenize(g.vocab, qwenGuardHead, false, true)
	g.tail = llama.Tokenize(g.vocab, qwenGuardTail, false, true)
	g.respHead = llama.Tokenize(g.vocab, qwenGuardRespHead, false, true)
	g.respMid = llama.Tokenize(g.vocab, qwenGuardRespMid, false, true)
	g.respTail = llama.Tokenize(g.vocab, qwenGuardRespTail, false, true)
	for i, v := range qwenVerdicts {
		t := llama.Tokenize(g.vocab, " "+v, false, false)
		if len(t) == 0 {
			g.Close()
			return nil, fmt.Errorf("laya: verdict %q does not tokenize", v)
		}
		g.verdict[i] = t[0]
		for _, prev := range g.verdict[:i] {
			if prev == t[0] {
				g.Close()
				return nil, fmt.Errorf("laya: verdicts share their first token (%d): not a Qwen3Guard vocab", prev)
			}
		}
	}
	if len(g.respHead)+len(g.respMid)+len(g.respTail) >= g.maxTokens {
		g.Close()
		return nil, fmt.Errorf("laya: MaxTokens %d is smaller than the %d token template", g.maxTokens, len(g.respHead)+len(g.respMid)+len(g.respTail))
	}

	for range contexts {
		cp := llama.ContextDefaultParams()
		cp.NCtx = uint32(g.maxTokens)
		cp.NBatch = uint32(g.maxTokens)
		cp.NUbatch = uint32(g.maxTokens)
		cp.NSeqMax = 1
		cp.NThreads, cp.NThreadsBatch = int32(o.Threads), int32(o.Threads)
		ctx, err := llama.InitFromModel(model, cp)
		if err != nil {
			g.Close()
			return nil, fmt.Errorf("laya: create context: %w", err)
		}
		g.pool <- &guardCtx{ctx: ctx}
		g.made++
	}
	return g, nil
}

// Close waits for running prompts, then releases the contexts and the model.
func (g *QwenGuard) Close() {
	for range g.made {
		llama.Free((<-g.pool).ctx)
	}
	g.made = 0
	if g.model != 0 {
		llama.ModelFree(g.model)
		g.model = 0
	}
}

// Predict answers every question from one forward pass over state.
func (g *QwenGuard) Predict(state string, questions []Question) ([]Result, error) {
	if len(questions) == 0 {
		return nil, errors.New("laya: no questions")
	}
	maps := make([][]int, len(questions)) // per question: option -> verdict index (Noul: nil)
	for i, q := range questions {
		m, err := qwenGuardOptions(q)
		if err != nil {
			return nil, err
		}
		maps[i] = m
	}
	lp, n, err := g.verdictLogProbs(state)
	if err != nil {
		return nil, err
	}
	return answerQuestions(questions, maps, lp, n), nil
}

// Answer answers questions from an existing prompt moderation (no second pass over the prompt).
func (g *QwenGuard) Answer(m Moderation, questions []Question) ([]Result, error) {
	var lp [3]float64
	for i, name := range qwenVerdicts {
		lp[i] = math.Log(math.Max(m.Probabilities[name], 1e-300))
	}
	maps := make([][]int, len(questions))
	for i, q := range questions {
		o, err := qwenGuardOptions(q)
		if err != nil {
			return nil, err
		}
		maps[i] = o
	}
	return answerQuestions(questions, maps, lp, m.Tokens), nil
}

func answerQuestions(questions []Question, maps [][]int, lp [3]float64, n int) []Result {
	results := make([]Result, len(questions))
	for i, q := range questions {
		var logits []float32
		if q.Kind == Noul {
			logits = []float32{float32(lp[0]), float32(logSumExp(lp[1], lp[2]))}
		} else {
			for _, v := range maps[i] {
				logits = append(logits, float32(lp[v]))
			}
		}
		results[i] = decodeLogits(q, logits, 1)
		results[i].Tokens = n
	}
	return results
}

// qwenGuardOptions maps a Choice question's options to verdict indexes.
func qwenGuardOptions(q Question) ([]int, error) {
	switch q.Kind {
	case Noul:
		return nil, nil
	case Choice:
		if len(q.Options) < 1 {
			return nil, fmt.Errorf("%w: question %q needs at least one option", ErrInput, q.ID)
		}
		idx := make([]int, len(q.Options))
		for i, o := range q.Options {
			idx[i] = -1
			for v, name := range qwenVerdicts {
				if strings.EqualFold(strings.TrimSpace(o.Label), name) {
					idx[i] = v
				}
			}
			if idx[i] < 0 {
				return nil, fmt.Errorf("%w: question %q: option %q is not one of safe, unsafe, controversial", ErrInput, q.ID, o.Label)
			}
		}
		return idx, nil
	}
	return nil, fmt.Errorf("%w: question %q: qwen3guard answers choice and noul questions only", ErrInput, q.ID)
}

// verdictLogProbs returns the log-probabilities of the three verdicts, normalized over the three,
// and the prompt length.
func (g *QwenGuard) verdictLogProbs(state string) ([3]float64, int, error) {
	lp, _, n, err := g.run(1, g.head, g.promptRest(state), 0, -1)
	return lp, n, err
}

// promptRest is the prompt-moderation template after its head, around state (untrusted: no special-token parsing).
func (g *QwenGuard) promptRest(state string) []llama.Token {
	ids := g.text(state)
	return append(ids[:len(ids):len(ids)], g.tail...)
}

// responseRest is the response-moderation template after its head, for one exchange.
func (g *QwenGuard) responseRest(prompt, response string) []llama.Token {
	var ids []llama.Token
	ids = append(ids, g.text(prompt)...)
	ids = append(ids, g.respMid...)
	ids = append(ids, g.text(response)...)
	return append(ids, g.respTail...)
}

func (g *QwenGuard) text(s string) []llama.Token {
	if s == "" {
		return nil
	}
	return llama.Tokenize(g.vocab, s, false, false)
}

// run decodes ids (a prompt ending in "Safety:"), returns the verdict log-probabilities, and then
// greedily generates up to maxGen more tokens (the first one is forced to the likeliest verdict),
// unless that verdict is skipGen (a verdict index, or -1).
func (g *QwenGuard) run(mode int, head, rest []llama.Token, maxGen, skipGen int) (lp [3]float64, gen []llama.Token, n int, err error) {
	n = len(head) + len(rest)
	if n+maxGen > g.maxTokens {
		return lp, nil, 0, fmt.Errorf("%w: input needs %d tokens, limit is %d", ErrInput, n+maxGen, g.maxTokens)
	}

	gc := <-g.pool
	ctx := gc.ctx
	defer func() { g.pool <- gc }()
	mem, err := llama.GetMemory(ctx)
	if err != nil {
		return lp, nil, 0, fmt.Errorf("laya: memory: %w", err)
	}
	// Keep the cached head, drop everything after it (the previous text and generation).
	start, prev := 0, gc.cached
	gc.cached = 0 // trusted again only once this request's prompt decode succeeded
	if prev == mode {
		start = len(head)
		if ok, err := llama.MemorySeqRm(mem, 0, llama.Pos(start), -1); err != nil || !ok {
			start = 0
		}
	}
	if start == 0 {
		if err := llama.MemoryClear(mem, true); err != nil {
			return lp, nil, 0, fmt.Errorf("laya: clear memory: %w", err)
		}
	}
	ids := append(head[:len(head):len(head)], rest...)[start:]

	batch := llama.BatchInit(int32(len(ids)), 0, 1)
	defer llama.BatchFree(batch)
	for i, id := range ids {
		if err := batch.Add(id, llama.Pos(start+i), []llama.SeqId{0}, i == len(ids)-1); err != nil {
			return lp, nil, 0, fmt.Errorf("laya: batch: %w", err)
		}
	}
	if rc, err := llama.Decode(ctx, batch); err != nil || rc != 0 {
		return lp, nil, 0, fmt.Errorf("laya: decode failed (rc=%d): %v", rc, err)
	}
	gc.cached = mode
	logits, err := llama.GetLogitsIth(ctx, -1, g.nVocab)
	if err != nil || len(logits) != g.nVocab {
		return lp, nil, 0, fmt.Errorf("laya: logits: got %d values, want %d (%v)", len(logits), g.nVocab, err)
	}

	var z [3]float64
	best := 0
	for i, t := range g.verdict {
		z[i] = float64(logits[t])
		if z[i] > z[best] {
			best = i
		}
	}
	lse := logSumExp(logSumExp(z[0], z[1]), z[2])
	for i := range z {
		lp[i] = z[i] - lse
	}

	if best == skipGen {
		return lp, nil, n, nil
	}
	tok := g.verdict[best]
	for len(gen) < maxGen {
		gen = append(gen, tok)
		if len(gen) == maxGen {
			break
		}
		one := []llama.Token{tok}
		if rc, err := llama.Decode(ctx, llama.BatchGetOne(one)); err != nil || rc != 0 {
			return lp, nil, 0, fmt.Errorf("laya: decode failed (rc=%d): %v", rc, err)
		}
		if logits, err = llama.GetLogitsIth(ctx, -1, g.nVocab); err != nil || len(logits) != g.nVocab {
			return lp, nil, 0, fmt.Errorf("laya: logits: got %d values, want %d (%v)", len(logits), g.nVocab, err)
		}
		tok = 0
		for t, l := range logits {
			if l > logits[tok] {
				tok = llama.Token(t)
			}
		}
		if llama.VocabIsEOG(g.vocab, tok) {
			break
		}
	}
	return lp, gen, n, nil
}

// Moderation is Qwen3Guard-Gen's full assessment of a prompt, or of a response to it.
type Moderation struct {
	Safety        string             `json:"safety"`            // Safe, Unsafe or Controversial
	Probabilities map[string]float64 `json:"probabilities"`     // the three verdicts, summing to 1
	Categories    []string           `json:"categories"`        // unsafe-content categories named (empty when safe)
	Refusal       *bool              `json:"refusal,omitempty"` // response moderation only: did the assistant refuse
	Tokens        int                `json:"-"`                 // input length
}

// Moderate assesses prompt (the user's query) when response is empty, else the response to it.
func (g *QwenGuard) Moderate(prompt, response string) (Moderation, error) {
	// A Safe prompt always lists "Categories: None", so that line is not generated (~150 ms);
	// a response still needs its refusal line.
	mode, head, rest, skip := 1, g.head, g.promptRest(prompt), 0
	if response != "" {
		mode, head, rest, skip = 2, g.respHead, g.responseRest(prompt, response), -1
	}
	lp, gen, n, err := g.run(mode, head, rest, qwenGuardMaxGen, skip)
	if err != nil {
		return Moderation{}, err
	}
	m := Moderation{Probabilities: map[string]float64{}, Tokens: n}
	best := 0
	for i, name := range qwenVerdicts {
		m.Probabilities[name] = math.Exp(lp[i])
		if lp[i] > lp[best] {
			best = i
		}
	}
	m.Safety = qwenVerdicts[best]
	m.Categories, m.Refusal = parseGuardText(llama.Detokenize(g.vocab, gen, false, false), response != "")
	return m, nil
}

// parseGuardText reads the generated "Safe\nCategories: ...\nRefusal: ..." (the leading "Safety:" was in the prompt).
func parseGuardText(text string, wantRefusal bool) (categories []string, refusal *bool) {
	_, rest, _ := strings.Cut(text, "Categories:")
	cats, tail, _ := strings.Cut(rest, "\n")
	categories = []string{}
	for _, c := range qwenGuardCategories {
		if strings.Contains(cats, c) {
			categories = append(categories, c)
		}
	}
	if wantRefusal {
		if _, v, ok := strings.Cut(tail, "Refusal:"); ok {
			r := strings.HasPrefix(strings.TrimSpace(v), "Yes")
			refusal = &r
		}
	}
	return categories, refusal
}

func logSumExp(a, b float64) float64 {
	m := math.Max(a, b)
	if math.IsInf(m, -1) {
		return m
	}
	return m + math.Log(math.Exp(a-m)+math.Exp(b-m))
}
