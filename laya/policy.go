package laya

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"unicode"

	"github.com/hybridgroup/yzma/pkg/llama"
)

// Targets say what kind of text the compiled questions will be asked about.
const (
	TargetRequest  = "request"  // a user message to an AI agent
	TargetResponse = "response" // an AI agent's reply or tool output
)

const (
	policyCtxTokens = 4096 // prompt + generation
	policyMaxGen    = 96   // tokens of one question
	policyCacheMax  = 64
)

// PolicyCompiler turns a written policy into System One yes/no questions with a small chat model
// (Qwen3-0.6B, GGUF via llama.cpp). Results are cached by policy, so the model runs once per policy.
type PolicyCompiler struct {
	model  llama.Model
	vocab  llama.Vocab
	nVocab int

	mu        sync.Mutex // one context: compilations run one at a time
	ctx       llama.Context
	cachedSys string // system prompt whose KV cache the context holds
	cache     map[[32]byte][]Question
	order     [][32]byte // cache keys, oldest first
}

// OpenPolicyCompiler loads the chat model at path. llama.cpp must be initialised first.
func OpenPolicyCompiler(path string, gpuLayers, threads int) (*PolicyCompiler, error) {
	if threads <= 0 {
		threads = max(1, runtime.GOMAXPROCS(0)/2)
	}
	mp := llama.ModelDefaultParams()
	mp.NGpuLayers = int32(gpuLayers)
	model, err := llama.ModelLoadFromFile(path, mp)
	if err != nil {
		return nil, fmt.Errorf("laya: load %s: %w", path, err)
	}
	cp := llama.ContextDefaultParams()
	cp.NCtx, cp.NBatch, cp.NUbatch, cp.NSeqMax = policyCtxTokens, policyCtxTokens, policyCtxTokens, 1
	cp.NThreads, cp.NThreadsBatch = int32(threads), int32(threads)
	ctx, err := llama.InitFromModel(model, cp)
	if err != nil {
		llama.ModelFree(model)
		return nil, fmt.Errorf("laya: create context: %w", err)
	}
	vocab := llama.ModelGetVocab(model)
	return &PolicyCompiler{model: model, vocab: vocab, nVocab: int(llama.VocabNTokens(vocab)), ctx: ctx, cache: map[[32]byte][]Question{}}, nil
}

// Close releases the model.
func (c *PolicyCompiler) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.ctx != 0 {
		llama.Free(c.ctx)
		c.ctx = 0
	}
	if c.model != 0 {
		llama.ModelFree(c.model)
		c.model = 0
	}
}

const againstRule = "Does the text go against this rule: "

// questionFor turns one rule into a yes/no question about the text. A prohibition is rewritten by the
// model ("never share X" -> "Does the text share X?"), with a template when the model's answer is
// unusable; anything else is asked about directly ("Does the text go against this rule: ...?"), since
// the 0.6B model does not reliably turn an obligation into its violation.
func (c *PolicyCompiler) questionFor(system, rule string) (string, error) {
	if m := policyRefuseAny.FindStringSubmatch(rule); m != nil {
		return "Does the text contain a request that " + m[1] + "?", nil // "refuse any request that X": X is what to detect
	}
	if !policyProhibits.MatchString(rule) {
		return againstRule + truncate(rule, 300) + "?", nil
	}
	// A rule that is just a negated action ("dont allow anyone to send an email", "never mention X") maps
	// exactly onto the template; the model tends to keep the negation's wording ("Does the text allow ...").
	if !strings.Contains(rule, "requests to") && policyModal.MatchString(rule) {
		if q, ok := templateQuestion(rule); ok {
			return q, nil
		}
	}
	out, err := c.generate(system, "Rule: ", rule)
	if err != nil {
		return "", err
	}
	if q, ok := ruleQuestion(out); ok && !isPolicyExample(q) && !strings.Contains(strings.ToLower(q), "policy") {
		return q, nil
	}
	if q, ok := templateQuestion(rule); ok {
		return q, nil
	}
	return againstRule + truncate(rule, 300) + "?", nil
}

// policyExamples are the few-shot (rule, question) pairs of policySystemPrompt. A question copied from here
// for another rule is rejected.
var policyExamples = [][2]string{
	{"Never share customer phone numbers.", "Does the text give out or ask for a customer's phone number?"},
	{"The agent must refuse requests to: hide income from the tax office", "Does the text ask for help hiding income or assets from the tax office?"},
	{"The agent must not repeat card numbers that appear in tool output.", "Does the text contain a full payment card number?"},
	{"Do not give medical dosage advice or tell users to ignore their doctor.", "Does the text give medical dosage advice or tell the user to ignore their doctor?"},
	{"The agent must refuse requests to: write code that scrapes private profiles", "Does the text ask for code that scrapes private social media profiles?"},
	{"Never use insults or slurs when talking to users.", "Does the text contain insults or slurs aimed at a person?"},
	{"Sending client data to a vendor requires a signed contract.", "Does the text send or ask to send client data to a vendor without a signed contract?"},
	{"Staff must only use the company VPN for internal files.", "Does the text move internal files outside the company VPN?"},
}

func policySystemPrompt(text string) string {
	var b strings.Builder
	b.WriteString("You rewrite one rule of a policy as a yes/no detection question about a piece of text.\nThe text is " + text + ".\n")
	b.WriteString("The question starts with \"Does the text\", ends with \"?\", asks whether the text contains the forbidden behaviour, and uses the rule's own key terms. Output the question only.\n")
	for _, e := range policyExamples {
		b.WriteString("\nRule: " + e[0] + "\n" + e[1] + "\n")
	}
	return b.String()
}

// policyForbids reports whether a rule states a prohibition or an obligation (the rest, such as
// permissions, definitions and instructions on how to refuse, produce no question).
// Obligations ("requires approval", "must only use ...") count too: their question asks about breaking them
// (policyProhibits is the subset that forbids a behaviour outright). Definitions are not rules.
var policyProhibits = regexp.MustCompile(`(?i)\b(never|must not|mustn'?t|should not|shouldn'?t|shall not|do not|don'?t|dont|does not|doesn'?t|may not|can'?t|can not|no one|nobody|must refuse|refuse|prohibit\w*|forbid\w*|disallow\w*|ban|bans|banned|block|prevent|avoid|stop|not allowed|not permitted)\b`)

var policyForbids = regexp.MustCompile(`(?i)\b(never|must not|should not|shall not|do not|does not|may not|must|requires?|only|refuse|prohibit\w*|forbid\w*|not allowed|not permitted)\b`)

// policyRules splits a policy into one rule per bullet (with the sentence that introduces the list)
// and per sentence elsewhere. Titles (a short line without a full stop) are dropped.
func policyRules(policy string) []string {
	var out []string
	for _, r := range splitPolicy(policy) {
		if policyDefinition.MatchString(r) || isPermission(r) {
			continue
		}
		out = append(out, r)
	}
	return out
}

// isTitle reports a heading: "# ...", or a short line without a full stop that is ALL CAPS or Title Case.
// A short lowercase line ("dont allow anyone to send an email") is a rule.
func isTitle(line string) bool {
	if strings.HasPrefix(line, "#") {
		return true
	}
	if len(line) >= 80 || strings.ContainsAny(line, ".!?") {
		return false
	}
	words := strings.Fields(line)
	upper, title := true, true
	for _, w := range words {
		r := []rune(w)
		if strings.ToUpper(w) != w {
			upper = false
		}
		if !unicode.IsUpper(r[0]) && !idStopwords[strings.ToLower(w)] {
			title = false
		}
	}
	return upper || title && len(words) > 1 && !policyProhibits.MatchString(line)
}

// isPermission reports a rule that only allows something ("the agent may discuss X"): nothing can break it.
// "Only managers can approve refunds" restricts, so it stays.
func isPermission(rule string) bool {
	return policyPermission.MatchString(rule) && !policyForbids.MatchString(rule)
}

var policyPermission = regexp.MustCompile(`(?i)\b(?:may|can|is (?:allowed|permitted|free) to|are (?:allowed|permitted|free) to|is fine)\b`)

// policyAndNot splits "be polite and never mention competitors" before the second rule. Go regexps have no
// lookahead, so the negation is matched as a separate clause start in splitAndNot.
var policyAndNot = andNotSplitter{regexp.MustCompile(`(?i)[,;]?\s+(?:and|but)\s+((?:never|don'?t|dont|do not|must not|should not|avoid|no)\b)`)}

type andNotSplitter struct{ re *regexp.Regexp }

// Split cuts s before each match, keeping the negation word with the clause that follows.
func (a andNotSplitter) Split(s string, _ int) []string {
	var out []string
	last := 0
	for _, m := range a.re.FindAllStringSubmatchIndex(s, -1) {
		out = append(out, s[last:m[0]])
		last = m[2] // start of the negation word
	}
	return append(out, s[last:])
}

var policyRefuseAny = regexp.MustCompile(`(?i)^refuse (?:any|every|all) requests? (?:that|which) (.+)$`)

// policyDefinition matches statements about the policy rather than rules in it: definitions, scope,
// consequences of violations and review notes.
var policyDefinition = regexp.MustCompile(`(?i)\b(?:is|are) (?:classified|defined|categori[sz]ed) as\b|\bthis policy (?:explains|describes|applies|covers|is reviewed)|\bit applies to\b` +
	`|^(?:this|these|that|it)\s+(?:includes?|covers|means)\b` +
	`|\bviolations?\b.*\b(?:may|will|can|could)\b.*\b(?:lead|result)\b|\bmay be (?:updated|changed|revised|amended)\b|\b(?:is|are) reviewed\b`)

func splitPolicy(policy string) []string {
	var rules []string
	lead := ""
	policy = policyMarkdown.Replace(policy)
	for _, line := range strings.Split(strings.ReplaceAll(policy, "\r\n", "\n"), "\n") {
		line = strings.TrimSpace(policyTitle.ReplaceAllString(strings.TrimSpace(line), ""))
		switch {
		case line == "":
			lead = ""
		case policyBullet.MatchString(line):
			if item := strings.TrimSpace(policyBullet.ReplaceAllString(line, "")); item != "" {
				rules = append(rules, strings.TrimSpace(lead+" "+item))
			}
		case strings.HasSuffix(line, ":"):
			lead = line
		case isTitle(line):
			// a heading, not a rule
		default:
			lead = ""
			for _, sent := range policySentence.Split(line, -1) {
				if sent = strings.TrimRight(strings.TrimSpace(sent), ".!? "); len(sent) > 3 {
					for _, part := range policyAndNot.Split(sent, -1) {
						if part = strings.TrimSpace(part); len(part) < 4 {
							continue
						}
						if strings.Contains(part, "requests to ") {
							rules = append(rules, splitRequests(part)...)
						} else {
							rules = append(rules, splitItems(part)...)
						}
					}
				}
			}
		}
	}
	return rules
}

// splitItems splits a clause like "rules require a, never b, verifying c, and never d" into one rule per
// item, each with the lead ("rules require") of the first. Commas inside parentheses, and commas that
// continue a list ("on email, devices, or platforms"), do not split: an item starts with a gerund
// ("verifying", "using") or a rule word ("never", "only").
func splitItems(clause string) []string {
	frags := splitTopLevel(clause)
	if len(frags) < 2 {
		return []string{clause}
	}
	// A gerund starts an item only in a list of requirements ("rules require sharing ..., verifying ...");
	// elsewhere it is just a noun in a list ("signing keys, session cookies").
	inList := policyLead.MatchString(frags[0])
	var items []string
	for _, f := range frags {
		f = strings.TrimPrefix(strings.TrimPrefix(strings.TrimSpace(f), "and "), "or ")
		first, _, _ := strings.Cut(strings.ToLower(f), " ")
		if len(items) == 0 || inList && strings.HasSuffix(first, "ing") || policyForbids.MatchString(first) {
			items = append(items, f)
		} else {
			items[len(items)-1] += ", " + f
		}
	}
	m := policyLead.FindStringSubmatch(items[0])
	if len(items) < 2 || m == nil {
		return items
	}
	rules := []string{items[0]}
	for _, it := range items[1:] {
		switch {
		case policyProhibits.MatchString(it):
			rules = append(rules, it) // "never sharing ..." stands on its own
		case policyForbids.MatchString(it):
			rules = append(rules, m[1]+": "+it) // "using only approved channels" needs its lead
		} // anything else ("verifying the identity of ...") is a process step the text cannot violate
	}
	return rules
}

// policyLead finds the part of a clause that introduces its list ("general rules require ...").
var policyLead = regexp.MustCompile(`(?i)^(.*?\b(?:requires?|must|should))\s+\S`)

// splitTopLevel splits on commas outside parentheses.
func splitTopLevel(s string) []string {
	var out []string
	depth, start := 0, 0
	for i, r := range s {
		switch r {
		case '(':
			depth++
		case ')':
			depth = max(0, depth-1)
		case ',':
			if depth == 0 {
				out = append(out, s[start:i])
				start = i + 1
			}
		}
	}
	return append(out, s[start:])
}

// splitRequests splits a run-on "... must refuse requests to a, b, or c" sentence into its prohibition
// (the part before) and one rule per listed request. Other sentences are returned as they are.
func splitRequests(sent string) []string {
	i := strings.Index(sent, "requests to ")
	if i < 0 {
		return []string{sent}
	}
	head := policyParenthetical.ReplaceAllString(sent[:i], " ")
	head = strings.TrimSpace(policyTrailingModal.ReplaceAllString(head, ""))
	lead := "The agent must refuse requests to:"
	var rules []string
	if len(head) > 12 {
		rules = append(rules, head)
	}
	// Items are separated by commas, but commas also separate nouns ("from memory, tools, or logs"):
	// a fragment starts a new item only when it begins with a verb.
	var items []string
	for _, frag := range policyListSplit.Split(sent[i+len("requests to "):], -1) {
		if frag = strings.TrimSpace(frag); frag == "" {
			continue
		}
		first, _, _ := strings.Cut(strings.ToLower(frag), " ")
		if len(items) == 0 || policyVerbs[first] {
			items = append(items, frag)
		} else {
			items[len(items)-1] += ", " + frag
		}
	}
	for _, item := range items {
		if len(item) > 3 {
			rules = append(rules, lead+" "+item)
		}
	}
	return rules
}

var (
	policyTitle         = regexp.MustCompile(`^[A-Z][A-Z0-9 &/,'-]{7,}:\s*`) // "SECRET POLICY: ..."
	policyParenthetical = regexp.MustCompile(`\s*[\x{2014}\x{2013}]\s*including[^\x{2014}\x{2013}]*[\x{2014}\x{2013}]`)
	policyTrailingModal = regexp.MustCompile(`(?i)[\s,]*(?:and\s+)?(?:must|should)(?:\s+refuse)?\s*$`)
	policyListSplit     = regexp.MustCompile(`,\s*(?:or\s+)?`)
	policyVerbs         = verbSet("reveal print display show list provide give share send write generate create access read extract bypass disclose expose use store log repeat transform encode decode decrypt place put help assist obtain tell upload download delete modify weaken leak copy paste dump return output")
	policyBullet        = regexp.MustCompile(`^\s*(?:[-*\x{2022}]|\d+[.)])\s+`)
	policySentence      = regexp.MustCompile(`[.!?;]\s+`)
	policyMarkdown      = strings.NewReplacer("**", "", "__", "", "`", "")
)

// Compile returns the questions for policy, about text of the given target (TargetRequest or
// TargetResponse). maxQuestions bounds the count.
func (c *PolicyCompiler) Compile(policy, target string, maxQuestions int) ([]Question, error) {
	if strings.TrimSpace(policy) == "" {
		return nil, fmt.Errorf("%w: policy is empty", ErrInput)
	}
	var text string
	switch target {
	case TargetRequest:
		text = "a message that a user sent to an AI agent"
	case TargetResponse:
		text = "an AI agent's reply or tool output"
	default:
		return nil, fmt.Errorf("%w: target must be %q or %q", ErrInput, TargetRequest, TargetResponse)
	}
	maxQuestions = max(1, min(maxQuestions, 32))
	key := sha256.Sum256(fmt.Appendf(nil, "%d\x00%s\x00%s", maxQuestions, target, policy))

	c.mu.Lock()
	defer c.mu.Unlock()
	if qs, ok := c.cache[key]; ok {
		return qs, nil
	}
	if c.ctx == 0 {
		return nil, errors.New("laya: policy compiler is closed")
	}
	rules := policyRules(policy)
	if len(rules) > 48 {
		return nil, fmt.Errorf("%w: policy has %d rules, max 48", ErrInput, len(rules))
	}
	system := policySystemPrompt(text)
	var qs []Question
	seen := map[string]bool{}
	add := func(rule string) error {
		if len(qs) == maxQuestions {
			return nil
		}
		q, err := c.questionFor(system, rule)
		if err != nil || q == "" {
			return err
		}
		base := questionID(q)
		if strings.HasPrefix(q, againstRule) {
			base = questionID(rule) // the generic wording would give every obligation the same id
		}
		id := base
		for n := 2; seen[id]; n++ {
			id = fmt.Sprintf("%s_%d", base, n)
		}
		seen[id] = true
		qs = append(qs, Question{ID: id, Kind: Noul, Instructions: q})
		return nil
	}
	// Any plain-English policy converts: every rule becomes a question, and a policy that did not split
	// into rules (a heading only, say) becomes one.
	if len(rules) == 0 {
		rules = []string{strings.TrimRight(strings.Join(strings.Fields(policyMarkdown.Replace(policy)), " "), ".!?;: ")}
	}
	for _, rule := range rules {
		if err := add(rule); err != nil {
			return nil, err
		}
	}
	if len(qs) == 0 {
		return nil, fmt.Errorf("%w: policy has no text to turn into questions", ErrInput)
	}
	if len(c.order) >= policyCacheMax { // drop the oldest entry, not the whole cache
		delete(c.cache, c.order[0])
		c.order = c.order[1:]
	}
	c.cache[key] = qs
	c.order = append(c.order, key)
	return qs, nil
}

// generate runs one chat turn greedily (Qwen3 chat template, thinking off). The system prompt is the
// same for every rule of a policy, so its KV cache is kept between turns. user is untrusted and
// tokenized without special-token parsing.
func (c *PolicyCompiler) generate(system, userHead, user string) (string, error) {
	tok := func(s string, special bool) []llama.Token { return llama.Tokenize(c.vocab, s, false, special) }
	head := tok("<|im_start|>system\n"+system+"<|im_end|>\n<|im_start|>user\n"+userHead, true)
	rest := append(tok(user, false), tok("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", true)...)
	if len(head)+len(rest)+policyMaxGen > policyCtxTokens {
		return "", fmt.Errorf("%w: rule needs %d tokens, limit is %d", ErrInput, len(head)+len(rest), policyCtxTokens-policyMaxGen)
	}

	mem, err := llama.GetMemory(c.ctx)
	if err != nil {
		return "", fmt.Errorf("laya: memory: %w", err)
	}
	start := 0
	if c.cachedSys == system {
		start = len(head)
		if ok, err := llama.MemorySeqRm(mem, 0, llama.Pos(start), -1); err != nil || !ok {
			start = 0
		}
	}
	c.cachedSys = ""
	if start == 0 {
		if err := llama.MemoryClear(mem, true); err != nil {
			return "", fmt.Errorf("laya: clear memory: %w", err)
		}
	}
	ids := append(head[:len(head):len(head)], rest...)[start:]
	batch := llama.BatchInit(int32(len(ids)), 0, 1)
	defer llama.BatchFree(batch)
	for i, id := range ids {
		if err := batch.Add(id, llama.Pos(start+i), []llama.SeqId{0}, i == len(ids)-1); err != nil {
			return "", fmt.Errorf("laya: batch: %w", err)
		}
	}
	if rc, err := llama.Decode(c.ctx, batch); err != nil || rc != 0 {
		return "", fmt.Errorf("laya: decode failed (rc=%d): %v", rc, err)
	}
	c.cachedSys = system // positions below len(head) hold this system prompt

	var gen []llama.Token
	for len(gen) < policyMaxGen {
		logits, err := llama.GetLogitsIth(c.ctx, -1, c.nVocab)
		if err != nil || len(logits) != c.nVocab {
			return "", fmt.Errorf("laya: logits: got %d values, want %d (%v)", len(logits), c.nVocab, err)
		}
		next := llama.Token(0)
		for t, l := range logits {
			if l > logits[next] {
				next = llama.Token(t)
			}
		}
		if llama.VocabIsEOG(c.vocab, next) {
			break
		}
		gen = append(gen, next)
		one := []llama.Token{next}
		if rc, err := llama.Decode(c.ctx, llama.BatchGetOne(one)); err != nil || rc != 0 {
			return "", fmt.Errorf("laya: decode failed (rc=%d): %v", rc, err)
		}
	}
	return llama.Detokenize(c.vocab, gen, false, false), nil
}

func isPolicyExample(q string) bool {
	for _, e := range policyExamples {
		if q == e[1] {
			return true
		}
	}
	return false
}

var policyModal = regexp.MustCompile(`(?i)^(?:(?:the agent|you|users?|staff|employees|people|anyone|no one|nobody)\s+)?` +
	`(?:must\s+never|must\s+not|mustn'?t|should\s+not|shouldn'?t|shall\s+not|never|do\s+not|don'?t|dont|does\s+not|doesn'?t|may\s+not|can'?t|can\s+not|cannot|may|can|should|must)\s+` +
	`(?:(?:allow|let|permit)\s+(?:anyone|anybody|users?|people|the agent|it|them)?\s*(?:to\s+)?)?`)

// templateQuestion builds a question for a rule without the model: "... requests to: <action>"
// asks for the action, any other prohibition asks whether the text does or requests it.
func templateQuestion(rule string) (string, bool) {
	if _, action, ok := strings.Cut(rule, "requests to:"); ok {
		if action = strings.TrimSpace(action); action != "" {
			return "Does the text ask the agent to " + action + "?", true
		}
		return "", false
	}
	action := strings.TrimSpace(policyModal.ReplaceAllString(rule, ""))
	if action == rule || len(action) < 6 {
		// An obligation ("requires written approval"): ask about breaking it.
		return againstRule + truncate(rule, 300) + "?", true
	}
	return "Does the text do, or ask someone to do, the following: " + action + "?", true
}

// ruleQuestion picks the "Does the text ...?" line out of the model's output.
func ruleQuestion(out string) (string, bool) {
	for line := range strings.SplitSeq(out, "\n") {
		line = strings.TrimSpace(line)
		if len(line) >= 20 && len(line) <= 600 && strings.HasPrefix(line, "Does the text") && strings.HasSuffix(line, "?") {
			return line, true
		}
	}
	return "", false
}

var idStopwords = map[string]bool{"a": true, "an": true, "the": true, "to": true, "for": true, "or": true, "and": true, "of": true,
	"in": true, "any": true, "ask": true, "asks": true, "text": true, "does": true, "agent": true, "user": true, "that": true,
	"such": true, "as": true, "do": true, "following": true, "general": true, "rules": true, "rule": true, "require": true, "requires": true, "from": true, "with": true, "their": true, "its": true, "is": true}

// questionID is a snake_case name from the first words of the question that carry meaning.
func questionID(q string) string {
	var words []string
	for _, w := range strings.FieldsFunc(strings.ToLower(q), func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) }) {
		if !idStopwords[w] {
			words = append(words, w)
		}
		if len(words) == 4 {
			break
		}
	}
	if len(words) == 0 {
		return "rule"
	}
	return strings.Join(words, "_")
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func verbSet(words string) map[string]bool {
	m := map[string]bool{}
	for _, w := range strings.Fields(words) {
		m[w] = true
	}
	return m
}
