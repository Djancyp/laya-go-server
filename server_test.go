package main

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"laya-server/laya"
)

type fakeModel struct {
	mu    sync.Mutex
	got   []laya.Question
	state string
	err   error
	block chan struct{} // if set, Predict waits for it
}

func (f *fakeModel) Predict(state string, qs []laya.Question) ([]laya.Result, error) {
	if f.block != nil {
		<-f.block
	}
	f.mu.Lock()
	f.state, f.got = state, qs
	f.mu.Unlock()
	if f.err != nil {
		return nil, f.err
	}
	rs := make([]laya.Result, len(qs))
	for i, q := range qs {
		r := laya.Result{ID: q.ID, Kind: q.Kind, Confidence: 0.9, AnswerConfidence: 0.8}
		switch q.Kind {
		case laya.Choice:
			r.Choice, r.Probs = q.Options[0].Label, make([]float64, len(q.Options))
			r.Probs[0] = 1
		case laya.Score:
			r.Score, r.Probs = 1.5, []float64{0.5, 0.5}
		case laya.Noul:
			r.Noul, r.Probs = 0.7, []float64{0.3, 0.7}
		}
		rs[i] = r
	}
	return rs, nil
}

func newTestServer(m predictor, keys ...string) *server {
	s := &server{
		model:         m,
		log:           slog.New(slog.NewTextHandler(io.Discard, nil)),
		limits:        limits{MaxQuestions: 4, MaxStateBytes: 100, MaxInstructions: 100},
		timeout:       time.Second,
		sem:           make(chan struct{}, 2),
		policySem:     make(chan struct{}, 2),
		policyTimeout: time.Second,
	}
	for _, k := range keys {
		s.keys = append(s.keys, sha256.Sum256([]byte(k)))
	}
	return s
}

func post(h http.Handler, auth, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(body))
	if auth != "" {
		req.Header.Set("Authorization", auth)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

const okBody = `{"state":"hi","questions":{
  "t":{"type":"choice","instructions":"which","criteria":{"zeta":"z","alpha":"a"}},
  "s":{"type":"score","instructions":"how big","criteria":["low","high"]},
  "n":{"type":"noul","instructions":"yes?"}}}`

func TestSystemOne(t *testing.T) {
	f := &fakeModel{}
	h := newTestServer(f).handler()
	rec := post(h, "", okBody)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	for _, want := range []string{`"choice":"zeta"`, `"score":1.5`, `"noul":0.7`, `"probabilities":{"alpha":0,"zeta":1}`, `"legend":{"0":"low","1":"high"}`, `"model":"laya-multilingual"`, `"usage":{"input_tokens":`} {
		if !strings.Contains(rec.Body.String(), want) {
			t.Errorf("body missing %s: %s", want, rec.Body)
		}
	}
	// questions sorted by id; choice option order is the request's document order
	if f.got[0].ID != "n" || f.got[2].ID != "t" || f.got[2].Options[0].Label != "zeta" {
		t.Errorf("order not preserved: %+v", f.got)
	}
}

func TestStateAcceptsJSONValue(t *testing.T) {
	f := &fakeModel{}
	post(newTestServer(f).handler(), "", `{"state":{"a": 1},"questions":{"n":{"type":"noul","instructions":"q"}}}`)
	if f.state != `{"a":1}` {
		t.Errorf("state = %q", f.state)
	}
}

func TestTypeSafeShapes(t *testing.T) {
	f := &fakeModel{}
	body := `{"state":{"m":"x"},"questions":{
	  "a":{"type":"score","instructions":{"about":"m","q":"how big?"},"criteria":["small","big"]},
	  "b":{"type":"choice","instructions":"which","criteria":{"x":null,"y":null}}}}`
	if rec := post(newTestServer(f).handler(), "", body); rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if f.got[0].Instructions != `{"about":"m","q":"how big?"}` || len(f.got[0].Options) != 2 || len(f.got[1].Options) != 2 {
		t.Errorf("parsed wrong: %+v", f.got)
	}
}

func TestAuth(t *testing.T) {
	h := newTestServer(&fakeModel{}, "k1", "k2").handler()
	for auth, want := range map[string]int{"": 401, "Bearer nope": 401, "k1": 401, "Bearer k1": 200, "Bearer k2": 200} {
		if got := post(h, auth, okBody).Code; got != want {
			t.Errorf("auth %q: %d, want %d", auth, got, want)
		}
	}
}

func TestValidation(t *testing.T) {
	h := newTestServer(&fakeModel{}).handler()
	cases := map[string]struct {
		body string
		code int
	}{
		"bad json":        {`{`, 400},
		"no state":        {`{"questions":{"n":{"type":"noul","instructions":"q"}}}`, 422},
		"no questions":    {`{"state":"x","questions":{}}`, 422},
		"long state":      {`{"state":"` + strings.Repeat("x", 101) + `","questions":{"n":{"type":"noul","instructions":"q"}}}`, 422},
		"bad type":        {`{"state":"x","questions":{"n":{"type":"nope","instructions":"q"}}}`, 422},
		"no instructions": {`{"state":"x","questions":{"n":{"type":"noul"}}}`, 422},
		"one option":      {`{"state":"x","questions":{"n":{"type":"choice","instructions":"q","criteria":{"a":"b"}}}}`, 422},
		"dup label":       {`{"state":"x","questions":{"n":{"type":"choice","instructions":"q","criteria":{"a":"1","a":"2"}}}}`, 422},
		"noul bad key":    {`{"state":"x","questions":{"n":{"type":"noul","instructions":"q","criteria":{"maybe":"?"}}}}`, 422},
		"score no levels": {`{"state":"x","questions":{"n":{"type":"score","instructions":"q"}}}`, 422},
		"score 11 levels": {`{"state":"x","questions":{"n":{"type":"score","instructions":"q","criteria":["1","2","3","4","5","6","7","8","9","10","11"]}}}`, 422},
	}
	for name, c := range cases {
		if got := post(h, "", c.body).Code; got != c.code {
			t.Errorf("%s: %d, want %d", name, got, c.code)
		}
	}
}

func TestModelErrors(t *testing.T) {
	for err, want := range map[error]int{
		fmt.Errorf("%w: too long", laya.ErrInput): 422,
		errors.New("decode failed"):               500,
	} {
		rec := post(newTestServer(&fakeModel{err: err}).handler(), "", okBody)
		if rec.Code != want {
			t.Errorf("%v: %d, want %d", err, rec.Code, want)
		}
		if want == 500 && strings.Contains(rec.Body.String(), "decode failed") {
			t.Error("internal error text leaked to client")
		}
	}
}

func TestOverloadAndTimeout(t *testing.T) {
	f := &fakeModel{block: make(chan struct{})}
	s := newTestServer(f)
	s.timeout = 50 * time.Millisecond
	h := s.handler()

	var wg sync.WaitGroup
	codes := make(chan int, 2)
	for range 2 { // fill both slots; each times out but keeps its slot while Predict blocks
		wg.Add(1)
		go func() { defer wg.Done(); codes <- post(h, "", okBody).Code }()
	}
	wg.Wait()
	close(codes)
	for c := range codes {
		if c != 504 {
			t.Errorf("blocked request: %d, want 504", c)
		}
	}
	if rec := post(h, "", okBody); rec.Code != 529 || rec.Header().Get("Retry-After") == "" {
		t.Errorf("full server: %d, want 529 with Retry-After", rec.Code)
	}

	close(f.block) // model unblocks: slots free up again
	deadline := time.Now().Add(time.Second)
	for post(h, "", okBody).Code != 200 {
		if time.Now().After(deadline) {
			t.Fatal("slots never freed")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestProbes(t *testing.T) {
	s := newTestServer(&fakeModel{}, "k")
	h := s.handler()
	get := func(p string) int {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("GET", p, nil))
		return rec.Code
	}
	if get("/healthz") != 200 || get("/readyz") != 503 {
		t.Error("before ready: want healthz 200, readyz 503")
	}
	s.ready.Store(true)
	if get("/readyz") != 200 {
		t.Error("after ready: want readyz 200")
	}
}

type panicModel struct{}

func (panicModel) Predict(string, []laya.Question) ([]laya.Result, error) { panic("boom") }

type shortModel struct{}

func (shortModel) Predict(string, []laya.Question) ([]laya.Result, error) { return nil, nil }

func TestModelPanicAndBadResult(t *testing.T) {
	for name, m := range map[string]predictor{"panic": panicModel{}, "short": shortModel{}} {
		t.Run(name, func(t *testing.T) {
			s := newTestServer(m)
			rec := post(s.handler(), "", okBody)
			if rec.Code != 500 {
				t.Fatalf("status %d: %s", rec.Code, rec.Body)
			}
			if len(s.sem) != 0 {
				t.Errorf("slot leaked: %d held", len(s.sem))
			}
		})
	}
}
