package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"laya-server/laya"
)

// policyCompiler turns policy text into questions (laya.PolicyCompiler; a fake in tests).
type policyCompiler interface {
	Compile(policy, target string, maxQuestions int) ([]laya.Question, error)
}

const maxPolicyBytes = 8 << 10

// policyRequest is the body of POST /v1/policy.
type policyRequest struct {
	Policy string `json:"policy"`
	Target string `json:"target"` // "request" (default): the questions are about a user message; "response": about an agent reply
}

// policyQuestion is one converted question in System One request form.
type policyQuestion struct {
	Type         string `json:"type"`
	Instructions string `json:"instructions"`
}

// policy converts a written policy to System One questions with the policy model and returns the
// {"questions"} to merge with a state and send to /v1/systemone (or to edit first).
func (s *server) policy(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)
	var req policyRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			writeError(w, http.StatusRequestEntityTooLarge, "body_too_large", "request body over 1 MiB")
			return
		}
		writeError(w, http.StatusBadRequest, "bad_json", err.Error())
		return
	}
	bad := func(msg string) { writeError(w, http.StatusUnprocessableEntity, "invalid_request", msg) }
	if req.Policy == "" || len(req.Policy) > maxPolicyBytes {
		bad(fmt.Sprintf("policy is required, max %d bytes", maxPolicyBytes))
		return
	}
	if req.Target == "" {
		req.Target = laya.TargetRequest
	}
	ph := s.policyC.Load()
	if ph == nil {
		if s.policyLoading.Load() {
			w.Header().Set("Retry-After", "30")
			writeError(w, http.StatusServiceUnavailable, "policy_model_loading", "the policy model is being downloaded or loaded, retry shortly")
			return
		}
		writeError(w, http.StatusNotImplemented, "not_configured", "policy endpoint is off (LAYA_POLICY_MODEL=off or the model failed to load)")
		return
	}

	select {
	case s.policySem <- struct{}{}:
	default:
		w.Header().Set("Retry-After", "5")
		writeError(w, 529, "overloaded", "too many policy requests in flight")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), s.policyTimeout)
	defer cancel()

	type outcome struct {
		qs  []laya.Question
		err error
	}
	done := make(chan outcome, 1)
	go func() {
		defer func() { <-s.policySem }()
		defer func() {
			if v := recover(); v != nil {
				s.log.Error("panic in policy", "value", v)
				done <- outcome{err: fmt.Errorf("panic: %v", v)}
			}
		}()
		qs, err := ph.c.Compile(req.Policy, req.Target, s.limits.MaxQuestions)
		done <- outcome{qs, err}
	}()

	select {
	case o := <-done:
		switch {
		case errors.Is(o.err, laya.ErrInput):
			bad(o.err.Error())
		case o.err != nil:
			s.log.Error("policy failed", "err", o.err)
			writeError(w, http.StatusInternalServerError, "internal", "policy evaluation failed")
		default:
			questions := make(map[string]policyQuestion, len(o.qs))
			for _, q := range o.qs {
				questions[q.ID] = policyQuestion{Type: "noul", Instructions: q.Instructions}
			}
			writeJSON(w, http.StatusOK, map[string]any{"questions": questions})
		}
	case <-ctx.Done():
		writeError(w, http.StatusGatewayTimeout, "timeout", "request timed out")
	}
}
