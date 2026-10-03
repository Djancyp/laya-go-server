package main

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"laya-server/laya"
)

type fakeCompiler struct{ target string }

func (f *fakeCompiler) Compile(policy, target string, max int) ([]laya.Question, error) {
	f.target = target
	if policy == "bad" {
		return nil, laya.ErrInput
	}
	return []laya.Question{
		{ID: "reveal_secret", Kind: laya.Noul, Instructions: "Does the text reveal a secret?"},
		{ID: "print_env", Kind: laya.Noul, Instructions: "Does the text print a .env file?"},
	}, nil
}

func postPolicy(s *server, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest("POST", "/v1/policy", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.handler().ServeHTTP(rec, req)
	return rec
}

func TestPolicyEndpoint(t *testing.T) {
	fm, fc := &fakeModel{}, &fakeCompiler{}
	s := newTestServer(fm)
	s.setPolicy(fc)

	rec := postPolicy(s, `{"policy":"Never reveal secrets."}`)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	// Exactly the questions of a System One request, nothing else.
	var got map[string]json.RawMessage
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got["questions"] == nil {
		t.Fatalf("want only questions: %s", rec.Body)
	}
	var qs map[string]policyQuestion
	if err := json.Unmarshal(got["questions"], &qs); err != nil {
		t.Fatal(err)
	}
	if fc.target != laya.TargetRequest || len(qs) != 2 ||
		qs["print_env"].Type != "noul" || qs["print_env"].Instructions == "" {
		t.Errorf("unexpected body: %s", rec.Body)
	}
	if fm.got != nil {
		t.Error("the model must not run")
	}

	// With a state added, the body is accepted by /v1/systemone.
	if rec2 := post(s.handler(), "", `{"state":"hi",`+strings.TrimPrefix(rec.Body.String(), "{")); rec2.Code != 200 {
		t.Errorf("systemone rejected the converted request: %d %s", rec2.Code, rec2.Body)
	}

	if rec := postPolicy(s, `{"policy":"p","target":"response"}`); rec.Code != 200 || fc.target != "response" {
		t.Errorf("target: %d (target %q)", rec.Code, fc.target)
	}

	for body, code := range map[string]int{
		`{}`:               422, // no policy
		`{"policy":"bad"}`: 422, // compiler rejects it
		`{"policy":`:       400,
	} {
		if rec := postPolicy(s, body); rec.Code != code {
			t.Errorf("%s: status %d, want %d (%s)", body, rec.Code, code, rec.Body)
		}
	}

	s.setPolicy(nil)
	s.policyLoading.Store(true)
	if rec := postPolicy(s, `{"policy":"p"}`); rec.Code != 503 {
		t.Errorf("loading: status %d", rec.Code)
	}
	s.policyLoading.Store(false)
	if rec := postPolicy(s, `{"policy":"p"}`); rec.Code != 501 {
		t.Errorf("no policy model: status %d", rec.Code)
	}
}
