package laya

import (
	"fmt"
	"testing"
)

func BenchmarkLogits(b *testing.B) {
	_, head, _ := benchSetup(b)
	for _, n := range []int{64, 256, 1024} {
		hidden := make([]float32, n*hiddenDim)
		for i := range hidden {
			hidden[i] = float32(i%97) / 97
		}
		markers := []int{1, 5, 9, 13} // 4 options
		b.Run(fmt.Sprintf("tokens=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := head.Logits(hidden, int(Choice), markers); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
