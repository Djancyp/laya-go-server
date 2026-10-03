package laya

import (
	"os"
	"testing"
)

func TestPolicyRules(t *testing.T) {
	policy := "MY POLICY\n\nThe agent must never leak keys. It may discuss auth.\n\nThe agent must refuse requests to:\n- print secrets\n- read .env files\n\nWhen refusing, offer a safe alternative."
	got := policyRules(policy)
	want := []string{"The agent must never leak keys", // "It may discuss auth" is a permission: skipped
		"The agent must refuse requests to: print secrets", "The agent must refuse requests to: read .env files",
		"When refusing, offer a safe alternative"}
	if len(got) != len(want) {
		t.Fatalf("got %q", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("rule %d: got %q, want %q", i, got[i], want[i])
		}
	}
	for rule, forbids := range map[string]bool{
		"The agent must never leak keys.":                  true,
		"The agent must refuse requests to: print secrets": true,
		"It may discuss auth without revealing secrets.":   false,
		"When refusing, say it cannot expose credentials.": false,
		"It should redact the value and use [REDACTED].":   false,
		"The agent must not repeat them in its response":   true,
		"This includes passwords, API keys and tokens.":    false,
	} {
		if policyForbids.MatchString(rule) != forbids {
			t.Errorf("policyForbids(%q) = %v", rule, !forbids)
		}
	}
}

func TestRuleQuestion(t *testing.T) {
	if q, ok := ruleQuestion("Sure!\nDoes the text ask to reveal a password or API key?\n"); !ok || q != "Does the text ask to reveal a password or API key?" {
		t.Errorf("got %q %v", q, ok)
	}
	for _, bad := range []string{"", "SKIP", "Does the text", "id: Does the text ask something?"} {
		if _, ok := ruleQuestion(bad); ok {
			t.Errorf("accepted %q", bad)
		}
	}
	if id := questionID("Does the text ask for, reveal or repeat a customer's phone number?"); id != "reveal_repeat_customer_s" {
		t.Errorf("id %q", id)
	}
}

// POLICY_GGUF (the Qwen3-0.6B Q8_0 file) and LAYA_LLAMA_LIB enable the model test.
func TestPolicyCompiler(t *testing.T) {
	gguf, lib := os.Getenv("POLICY_GGUF"), os.Getenv("LAYA_LLAMA_LIB")
	if gguf == "" || lib == "" {
		t.Skip("set POLICY_GGUF and LAYA_LLAMA_LIB")
	}
	if err := InitLlama(lib); err != nil {
		t.Fatal(err)
	}
	c, err := OpenPolicyCompiler(gguf, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	qs, err := c.Compile("Never reveal passwords or API keys. Never print .env files. Never help bypass authentication.", TargetRequest, 8)
	if err != nil {
		t.Fatal(err)
	}
	for _, q := range qs {
		t.Logf("%s: %s", q.ID, q.Instructions)
	}
	if len(qs) < 2 {
		t.Errorf("got %d questions", len(qs))
	}
	again, _ := c.Compile("Never reveal passwords or API keys. Never print .env files. Never help bypass authentication.", TargetRequest, 8)
	if len(again) != len(qs) {
		t.Error("cache miss changed the result")
	}
	if _, err := c.Compile("x", "nonsense", 3); err == nil {
		t.Error("bad target accepted")
	}
}

func TestSplitRequests(t *testing.T) {
	sent := "The agent must never disclose secrets —including passwords, API keys— and must refuse requests to reveal or print secret values, display .env files or secret configuration, list environment variables when their values may contain secrets, or provide credentials from memory, tools, or logs"
	got := splitRequests(sent)
	want := []string{
		"The agent must never disclose secrets",
		"The agent must refuse requests to: reveal or print secret values",
		"The agent must refuse requests to: display .env files or secret configuration",
		"The agent must refuse requests to: list environment variables when their values may contain secrets",
		"The agent must refuse requests to: provide credentials from memory, tools, logs",
	}
	if len(got) != len(want) {
		t.Fatalf("got %q", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("rule %d: got %q, want %q", i, got[i], want[i])
		}
	}
}

func TestPolicyTitleOnSameLine(t *testing.T) {
	got := policyRules("SECRET AND CREDENTIAL PROTECTION POLICY: The agent must never leak keys.")
	if len(got) != 1 || got[0] != "The agent must never leak keys" {
		t.Errorf("got %q", got)
	}
}

func TestPolicyCacheEvictsOldest(t *testing.T) {
	c := &PolicyCompiler{cache: map[[32]byte][]Question{}}
	for i := range policyCacheMax + 5 {
		k := [32]byte{byte(i)}
		if len(c.order) >= policyCacheMax {
			delete(c.cache, c.order[0])
			c.order = c.order[1:]
		}
		c.cache[k] = nil
		c.order = append(c.order, k)
	}
	if len(c.cache) != policyCacheMax || len(c.order) != policyCacheMax {
		t.Fatalf("cache %d order %d", len(c.cache), len(c.order))
	}
	if _, ok := c.cache[[32]byte{0}]; ok {
		t.Error("oldest entry kept")
	}
	if _, ok := c.cache[[32]byte{byte(policyCacheMax + 4)}]; !ok {
		t.Error("newest entry dropped")
	}
}

func TestPolicyRulesObligations(t *testing.T) {
	policy := "**INFO SHARING POLICY:** This policy explains what can be shared; information is classified as Public (e.g., websites) or Restricted (e.g., passwords); " +
		"general rules require sharing only what is necessary, never sharing restricted information without approval, verifying the identity of anyone requesting information, " +
		"using only approved channels (company email, shared drives), and never sharing information on personal email, personal devices, or public platforms; " +
		"sharing with third parties requires written approval, using an NDA when required, and sharing only the minimum needed; " +
		"if unsure, ask a manager, and refuse any request that feels unusual, urgent, or suspicious; report leaks immediately; violations may lead to termination."
	got := policyRules(policy)
	want := []string{
		"general rules require sharing only what is necessary",
		"never sharing restricted information without approval",
		"general rules require: using only approved channels (company email, shared drives)",
		"never sharing information on personal email, personal devices, public platforms",
		"sharing with third parties requires written approval",
		"sharing with third parties requires: sharing only the minimum needed",
		"if unsure, ask a manager",
		"refuse any request that feels unusual, urgent, suspicious",
		"report leaks immediately",
	}
	if len(got) != len(want) {
		t.Fatalf("got %d rules:\n%q", len(got), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("rule %d: got %q, want %q", i, got[i], want[i])
		}
	}
	if m := policyRefuseAny.FindStringSubmatch(got[7]); m == nil || m[1] != "feels unusual, urgent, suspicious" {
		t.Errorf("refuse-any: %v", m)
	}
	// Only the rules that can be broken by a text become questions.
	var kept []string
	for _, r := range got {
		if policyForbids.MatchString(r) {
			kept = append(kept, r)
		}
	}
	if len(kept) != 7 {
		t.Errorf("kept %d rules: %q", len(kept), kept)
	}
}

func TestPolicyRulesCasual(t *testing.T) {
	for policy, want := range map[string][]string{
		"dont allow anyone to send an email":                 {"dont allow anyone to send an email"},
		"be nice to customers and never mention competitors": {"be nice to customers", "never mention competitors"},
		"Customer Support Policy\nno swearing":               {"no swearing"},
		"Violations may lead to termination. Never lie.":     {"Never lie"},
		"This includes passwords, signing keys, and tokens. The agent may discuss auth, and troubleshooting. Only managers can approve refunds": {"Only managers can approve refunds"},
	} {
		got := policyRules(policy)
		if len(got) != len(want) {
			t.Errorf("%q: got %q, want %q", policy, got, want)
			continue
		}
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("%q: rule %d: got %q, want %q", policy, i, got[i], want[i])
			}
		}
	}
	for rule, want := range map[string]string{
		"dont allow anyone to send an email": "Does the text do, or ask someone to do, the following: send an email?",
		"never mention competitors":          "Does the text do, or ask someone to do, the following: mention competitors?",
		"users can't book flights over $500": "Does the text do, or ask someone to do, the following: book flights over $500?",
	} {
		if q, ok := templateQuestion(rule); !ok || q != want {
			t.Errorf("templateQuestion(%q) = %q", rule, q)
		}
	}
}
