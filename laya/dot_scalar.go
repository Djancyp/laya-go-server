//go:build !(goexperiment.simd && amd64)

package laya

// dot is the head's inner product. Build with GOEXPERIMENT=simd on amd64 for the AVX2 path.
func dot(a, b []float32) float32 { return dotScalar(a, b) }

// axpy is dst += alpha * x, attention's value accumulation.
func axpy(dst, x []float32, alpha float32) { axpyScalar(dst, x, alpha) }
