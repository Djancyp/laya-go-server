package main

import (
	"encoding/json"
	"fmt"
	"testing"

	"laya-server/laya"
)

func benchRequest(n int) request {
	body := `{"state":"Customer reports the invoice total is wrong.","questions":{`
	for i := range n {
		if i > 0 {
			body += ","
		}
		body += fmt.Sprintf(`"q%02d":{"type":"choice","instructions":"Which topic?","criteria":{"billing":"payments","tech":"bugs","sales":"pricing","other":""}}`, i)
	}
	var r request
	if err := json.Unmarshal([]byte(body+"}}"), &r); err != nil {
		panic(err)
	}
	return r
}

func BenchmarkParse(b *testing.B) {
	l := limits{MaxQuestions: 64, MaxStateBytes: 64 << 10, MaxInstructions: 4 << 10}
	for _, n := range []int{1, 16, 64} {
		r := benchRequest(n)
		b.Run(fmt.Sprintf("questions=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, _, err := r.parse(l); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkToAnswers(b *testing.B) {
	for _, n := range []int{1, 16, 64} {
		r := benchRequest(n)
		_, qs, _ := r.parse(limits{MaxQuestions: 64, MaxStateBytes: 1 << 10, MaxInstructions: 1 << 10})
		rs := make([]laya.Result, len(qs))
		for i, q := range qs {
			rs[i] = laya.Result{ID: q.ID, Kind: laya.Choice, Choice: "tech", Confidence: 0.9, Probs: []float64{0.1, 0.7, 0.1, 0.1}}
		}
		b.Run(fmt.Sprintf("questions=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				toAnswers(qs, rs)
			}
		})
	}
}
