package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"

	"laya-server/laya"
)

// TypeSafe's documented option limits.
const (
	maxChoices = 255
	maxLevels  = 10
)

// limits bound what one request may cost. The model truncates to 1024 tokens anyway;
// these reject abuse before any tokenizing happens.
type limits struct {
	MaxQuestions    int
	MaxStateBytes   int
	MaxInstructions int
}

// request mirrors the TypeSafe System One shape: {"state": ..., "questions": {id: def}}.
type request struct {
	State     json.RawMessage `json:"state"`
	Questions map[string]qDef `json:"questions"`
}

type qDef struct {
	Type         string          `json:"type"`
	Instructions json.RawMessage `json:"instructions"` // text, or any JSON (sent as compact JSON text)
	// choice: {label: description}; noul: {"true": ..., "false": ...}; score: [level description, ...]
	Criteria json.RawMessage `json:"criteria"`
}

type kv struct{ Key, Val string }

// ordered is a JSON object that keeps document order: option order is part of the model input.
type ordered []kv

func (o *ordered) UnmarshalJSON(b []byte) error {
	dec := json.NewDecoder(bytes.NewReader(b))
	if t, err := dec.Token(); err != nil || t != json.Delim('{') {
		return errors.New("must be an object of strings")
	}
	seen := map[string]bool{}
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return err
		}
		var v string
		if err := dec.Decode(&v); err != nil {
			return errors.New("values must be strings")
		}
		k, ok := kt.(string)
		if !ok {
			return errors.New("must be an object of strings")
		}
		if seen[k] {
			return fmt.Errorf("duplicate key %q", k)
		}
		seen[k] = true
		*o = append(*o, kv{k, v})
	}
	return nil
}

// parse validates the request and returns the state text and the questions in id order.
func (r *request) parse(l limits) (string, []laya.Question, error) {
	state, err := stateText(r.State)
	if err != nil {
		return "", nil, err
	}
	if len(state) > l.MaxStateBytes {
		return "", nil, fmt.Errorf("state is %d bytes, max %d", len(state), l.MaxStateBytes)
	}
	if n := len(r.Questions); n == 0 || n > l.MaxQuestions {
		return "", nil, fmt.Errorf("questions: got %d, want 1..%d", n, l.MaxQuestions)
	}

	ids := make([]string, 0, len(r.Questions))
	for id := range r.Questions {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	qs := make([]laya.Question, 0, len(ids))
	for _, id := range ids {
		q, err := r.Questions[id].toQuestion(id, l)
		if err != nil {
			return "", nil, fmt.Errorf("question %q: %w", id, err)
		}
		qs = append(qs, q)
	}
	return state, qs, nil
}

// stateText accepts a JSON string as-is; any other JSON value is sent as its compact JSON text.
func stateText(raw json.RawMessage) (string, error) {
	if isNull(raw) {
		return "", errors.New("state is required")
	}
	return jsonText(raw)
}

func isNull(raw json.RawMessage) bool { return len(raw) == 0 || bytes.Equal(raw, []byte("null")) }

func jsonText(raw json.RawMessage) (string, error) {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s, nil
	}
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		return "", errors.New("not valid JSON")
	}
	return buf.String(), nil
}

func (d qDef) toQuestion(id string, l limits) (laya.Question, error) {
	q := laya.Question{ID: id}
	if isNull(d.Instructions) {
		return q, errors.New("instructions are required")
	}
	ins, err := jsonText(d.Instructions)
	if err != nil {
		return q, fmt.Errorf("instructions: %w", err)
	}
	if len(ins) > l.MaxInstructions {
		return q, fmt.Errorf("instructions are %d bytes, max %d", len(ins), l.MaxInstructions)
	}
	q.Instructions = ins

	var criteria ordered // choice/noul: object, document order kept
	if d.Type != "score" && !isNull(d.Criteria) {
		if err := json.Unmarshal(d.Criteria, &criteria); err != nil {
			return q, fmt.Errorf("criteria: %w", err)
		}
	}

	switch d.Type {
	case "choice":
		if n := len(criteria); n < 2 || n > maxChoices {
			return q, fmt.Errorf("choice needs 2..%d criteria, got %d", maxChoices, n)
		}
		q.Kind = laya.Choice
		for _, c := range criteria {
			q.Options = append(q.Options, laya.Option{Label: c.Key, Description: c.Val})
		}
	case "score":
		var levels []string
		if err := json.Unmarshal(d.Criteria, &levels); err != nil {
			return q, errors.New("score criteria must be an array of strings")
		}
		if n := len(levels); n < 2 || n > maxLevels {
			return q, fmt.Errorf("score needs 2..%d levels, got %d", maxLevels, n)
		}
		q.Kind = laya.Score
		for _, desc := range levels {
			q.Options = append(q.Options, laya.Option{Description: desc})
		}
	case "noul":
		q.Kind = laya.Noul
		if len(criteria) > 0 { // optional: {"false": "...", "true": "..."}
			var no, yes string
			for _, c := range criteria {
				switch c.Key {
				case "false":
					no = c.Val
				case "true":
					yes = c.Val
				default:
					return q, fmt.Errorf(`noul criteria keys must be "false" and "true", got %q`, c.Key)
				}
			}
			q.Options = []laya.Option{{Description: no}, {Description: yes}}
		}
	default:
		return q, fmt.Errorf(`type must be "choice", "score" or "noul", got %q`, d.Type)
	}
	return q, nil
}

// answer follows the TypeSafe System One response: a noul has only its probability,
// choice and score add confidence and per-option probabilities (score also a legend).
type answer struct {
	Type          string             `json:"type"`
	Choice        *string            `json:"choice,omitempty"`
	Score         *float64           `json:"score,omitempty"`
	Noul          *float64           `json:"noul,omitempty"`
	Confidence    *float64           `json:"confidence,omitempty"`
	Legend        map[string]string  `json:"legend,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
}

type usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// toAnswers converts model results (same order as qs) to the response body and token usage.
func toAnswers(qs []laya.Question, rs []laya.Result) (map[string]answer, usage, error) {
	if len(rs) != len(qs) {
		return nil, usage{}, fmt.Errorf("model returned %d results for %d questions", len(rs), len(qs))
	}
	out := make(map[string]answer, len(rs))
	var u usage
	for i, r := range rs {
		if r.Kind != laya.Noul && len(r.Probs) != len(qs[i].Options) {
			return nil, usage{}, fmt.Errorf("question %q: %d probabilities for %d options", r.ID, len(r.Probs), len(qs[i].Options))
		}
		u.InputTokens += r.Tokens
		a := answer{Type: r.Kind.String()}
		switch r.Kind {
		case laya.Choice:
			a.Choice, a.Confidence = &r.Choice, &r.Confidence
			a.Probabilities = make(map[string]float64, len(r.Probs))
			for j, p := range r.Probs {
				a.Probabilities[qs[i].Options[j].Label] = p
			}
		case laya.Score:
			a.Score, a.Confidence = &r.Score, &r.Confidence
			a.Legend = make(map[string]string, len(r.Probs))
			a.Probabilities = make(map[string]float64, len(r.Probs))
			for j, p := range r.Probs {
				k := strconv.Itoa(j)
				a.Legend[k], a.Probabilities[k] = qs[i].Options[j].Description, p
			}
		case laya.Noul:
			a.Noul = &r.Noul
		}
		out[r.ID] = a
	}
	return out, u, nil
}
