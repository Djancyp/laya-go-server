//go:build goexperiment.simd && amd64

package laya

import "simd/archsimd"

// useSIMD gates the AVX2+FMA path; other amd64 CPUs fall back to dotScalar.
var useSIMD = archsimd.X86.AVX2() && archsimd.X86.FMA()

// dot is the head's inner product: ~89% of the head's CPU time is spent here. The AVX2
// path keeps four 8-wide FMA accumulators in flight to hide FMA latency; hidden sizes
// (768) and head slices (64) are multiples of 32, so the tails are rarely taken.
func dot(a, b []float32) float32 {
	if !useSIMD {
		return dotScalar(a, b)
	}
	n := len(a)
	b = b[:n]
	var acc0, acc1, acc2, acc3 archsimd.Float32x8
	i := 0
	for ; i+32 <= n; i += 32 {
		acc0 = archsimd.LoadFloat32x8(a[i:]).MulAdd(archsimd.LoadFloat32x8(b[i:]), acc0)
		acc1 = archsimd.LoadFloat32x8(a[i+8:]).MulAdd(archsimd.LoadFloat32x8(b[i+8:]), acc1)
		acc2 = archsimd.LoadFloat32x8(a[i+16:]).MulAdd(archsimd.LoadFloat32x8(b[i+16:]), acc2)
		acc3 = archsimd.LoadFloat32x8(a[i+24:]).MulAdd(archsimd.LoadFloat32x8(b[i+24:]), acc3)
	}
	for ; i+8 <= n; i += 8 {
		acc0 = archsimd.LoadFloat32x8(a[i:]).MulAdd(archsimd.LoadFloat32x8(b[i:]), acc0)
	}
	v := acc0.Add(acc1).Add(acc2.Add(acc3))
	q := v.GetLo().Add(v.GetHi())
	s := (q.GetElem(0) + q.GetElem(1)) + (q.GetElem(2) + q.GetElem(3))
	for ; i < n; i++ {
		s += a[i] * b[i]
	}
	return s
}

// axpy is dst += alpha * x: attention's value accumulation, the head's top cost at long
// sequences once dot is vectorized. Head slices are 64 wide, so the tail is rarely taken.
func axpy(dst, x []float32, alpha float32) {
	if !useSIMD {
		axpyScalar(dst, x, alpha)
		return
	}
	n := len(dst)
	x = x[:n]
	av := archsimd.BroadcastFloat32x8(alpha)
	i := 0
	for ; i+8 <= n; i += 8 {
		av.MulAdd(archsimd.LoadFloat32x8(x[i:]), archsimd.LoadFloat32x8(dst[i:])).Store(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] += alpha * x[i]
	}
}
