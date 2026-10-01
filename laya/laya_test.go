package laya

import (
	"bytes"
	"encoding/json"
	"math"
	"slices"
	"strings"
	"testing"
)

// orderedKeys returns the keys of a JSON object in document order (Go maps lose it, and
// option order is part of the model input).
func orderedKeys(t *testing.T, raw json.RawMessage) ([]string, map[string]json.RawMessage) {
	t.Helper()
	dec := json.NewDecoder(bytes.NewReader(raw))
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		t.Fatalf("not an object: %s", raw)
	}
	var keys []string
	vals := map[string]json.RawMessage{}
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			t.Fatal(err)
		}
		var v json.RawMessage
		if err := dec.Decode(&v); err != nil {
			t.Fatal(err)
		}
		k := kt.(string)
		keys = append(keys, k)
		vals[k] = v
	}
	return keys, vals
}

func str(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) != nil {
		return ""
	}
	return s
}

// refQuestions converts the reference file's question dicts to Questions, keeping key order.
func refQuestions(t *testing.T, raw map[string]json.RawMessage, only map[string]refQuestion) []Question {
	t.Helper()
	var qs []Question
	for id := range only {
		var def struct {
			Type         string          `json:"type"`
			Instructions string          `json:"instructions"`
			Criteria     json.RawMessage `json:"criteria"`
		}
		if err := json.Unmarshal(raw[id], &def); err != nil {
			t.Fatal(err)
		}
		q := Question{ID: id, Instructions: def.Instructions}
		switch def.Type {
		case "choice":
			q.Kind = Choice
			keys, vals := orderedKeys(t, def.Criteria)
			for _, k := range keys {
				q.Options = append(q.Options, Option{Label: k, Description: str(vals[k])})
			}
		case "score":
			q.Kind = Score
			var levels []string
			if err := json.Unmarshal(def.Criteria, &levels); err != nil {
				t.Fatal(err)
			}
			for _, l := range levels {
				q.Options = append(q.Options, Option{Description: l})
			}
		case "noul":
			q.Kind = Noul
			if len(def.Criteria) > 0 && string(def.Criteria) != "null" { // optional {"false": ..., "true": ...}
				_, vals := orderedKeys(t, def.Criteria)
				q.Options = []Option{{Label: "false", Description: str(vals["false"])}, {Label: "true", Description: str(vals["true"])}}
			}
		}
		qs = append(qs, q)
	}
	return qs
}

// refState returns the state as the exact text Python fed the model, or false for list
// states (conversation lists are left-truncated in Python; not ported).
func refState(raw json.RawMessage) (string, bool) {
	switch {
	case len(raw) > 0 && raw[0] == '"':
		return str(raw), true
	case len(raw) > 0 && raw[0] == '{':
		return string(raw), true // Python json.dumps(state, ensure_ascii=False) form
	}
	return "", false
}

func TestBuildSequenceMatchesPython(t *testing.T) {
	loadRef(t)
	tok := testTokenizer(t)
	_, cfg := testHead(t)
	m := &Model{tok: tok, cfg: *cfg}

	for ci, c := range refCases {
		state, ok := refState(c.State)
		if !ok {
			continue
		}
		stateIDs := tok.Encode(strings.ReplaceAll(state, tok.MaskText, " "))

		for _, q := range refQuestions(t, c.Questions, c.PerQuestion) {
			t.Run(q.ID+"/case"+string(rune('a'+ci)), func(t *testing.T) {
				ids, markers, err := m.buildSequence(q, stateIDs)
				if err != nil {
					t.Fatal(err)
				}
				want := c.PerQuestion[q.ID]
				if !slices.Equal(ids, want.IDs) {
					t.Errorf("ids differ (got %d tokens, want %d)\n got %v\nwant %v", len(ids), len(want.IDs), ids, want.IDs)
				}
				if !slices.Equal(markers, want.Markers) {
					t.Errorf("markers = %v, want %v", markers, want.Markers)
				}
			})
		}
	}
}

func TestPredictMatchesPython(t *testing.T) {
	loadRef(t)
	enc := testEncoder(t) // F16 recommended: LAYA_GGUF=laya-multilingual-F16.gguf
	tok := testTokenizer(t)
	head, cfg := testHead(t)
	m := &Model{tok: tok, enc: enc, head: head, cfg: *cfg}

	for ci, c := range refCases {
		state, ok := refState(c.State)
		if !ok {
			continue
		}
		qs := refQuestions(t, c.Questions, c.PerQuestion)
		got, err := m.Predict(state, qs)
		if err != nil {
			t.Fatal(err)
		}

		for i, r := range got {
			want := c.Answers[r.ID]
			t.Run(r.ID+"/case"+string(rune('a'+ci)), func(t *testing.T) {
				tol := 0.11 // quantized GGUFs drift; F16 (the default) is within 0.005
				if strings.Contains(envOr("LAYA_GGUF", "F16"), "F16") {
					tol = 0.01
				}
				switch r.Kind {
				case Choice:
					probs, _ := want["probabilities"].(map[string]any)
					for j, o := range qs[indexOf(qs, r.ID)].Options {
						if d := math.Abs(r.Probs[j] - probs[o.Label].(float64)); d > tol {
							t.Errorf("P(%s) = %.4f, python %.4f", o.Label, r.Probs[j], probs[o.Label])
						}
					}
				case Noul:
					if d := math.Abs(r.Noul - want["noul"].(float64)); d > tol {
						t.Errorf("noul = %.4f, python %.4f", r.Noul, want["noul"])
					}
				case Score:
					if d := math.Abs(r.Score - want["score"].(float64)); d > 0.5 {
						t.Errorf("score = %.4f, python %.4f", r.Score, want["score"])
					}
				}
				_ = i
			})
		}
	}
}

func indexOf(qs []Question, id string) int {
	for i, q := range qs {
		if q.ID == id {
			return i
		}
	}
	return -1
}
