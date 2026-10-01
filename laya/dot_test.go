package laya

import (
	"math"
	"testing"
)

// TestDotMatchesScalar pins dot and axpy (AVX2 under GOEXPERIMENT=simd) to the scalar references
// across lengths that hit the 32-wide loop, the 8-wide loop and the scalar tail.
func TestDotMatchesScalar(t *testing.T) {
	for _, n := range []int{0, 1, 7, 8, 9, 31, 32, 33, 64, 100, 768, 1152} {
		a, b := make([]float32, n), make([]float32, n+3) // b longer than a: dot reads len(a)
		for i := range a {
			a[i] = float32(i%11)/11 - 0.4
		}
		for i := range b {
			b[i] = float32(i%5)/5 - 0.6
		}
		got, want := dot(a, b), dotScalar(a, b)
		if d := math.Abs(float64(got - want)); d > 1e-4*math.Max(1, math.Abs(float64(want))) {
			t.Errorf("n=%d: dot = %v, scalar %v", n, got, want)
		}

		gotA, wantA := append([]float32(nil), a...), append([]float32(nil), a...)
		axpy(gotA, b, 0.37)
		axpyScalar(wantA, b, 0.37)
		for i := range gotA {
			if d := math.Abs(float64(gotA[i] - wantA[i])); d > 1e-6 {
				t.Fatalf("n=%d: axpy[%d] = %v, scalar %v", n, i, gotA[i], wantA[i])
			}
		}
	}
}
