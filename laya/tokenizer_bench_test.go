package laya

import (
	"fmt"
	"strings"
	"testing"
)

func BenchmarkEncode(b *testing.B) {
	tok, _, _ := benchSetup(b)
	for _, words := range []int{16, 256, 4096} {
		text := strings.Repeat("The quick brown fox jumps over the lazy dog. ", words/9+1)
		b.Run(fmt.Sprintf("words=%d", words), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				tok.Encode(text)
			}
			b.ReportMetric(float64(len(text))*float64(b.N)/b.Elapsed().Seconds()/1e6, "MB/s")
		})
	}
}
