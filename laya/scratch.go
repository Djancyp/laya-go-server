package laya

import "sync"

// arena is a bump allocator for the head's per-call temporaries. Logits used to allocate
// ~54 MB of fresh buffers per call at 1024 tokens; with a pooled arena that is reset between
// layer phases, steady-state calls allocate almost nothing and each in-flight call holds one
// phase's worth of scratch.
//
// Buffers from get have undefined contents and are valid only until the next reset. A nil
// *arena falls back to make, for results that must outlive the call.
type arena struct {
	buf []float32
	off int
}

func (a *arena) get(n int) []float32 {
	if a == nil {
		return make([]float32, n)
	}
	if a.off+n > len(a.buf) {
		// Slices already handed out keep the old slab alive until they die; the larger
		// slab then covers this whole phase from the next reset on.
		a.buf = make([]float32, max(2*len(a.buf), a.off+n))
		a.off = 0
	}
	s := a.buf[a.off : a.off+n : a.off+n]
	a.off += n
	return s
}

func (a *arena) reset() { a.off = 0 }

// scratch is one Logits call's reusable memory: the residual stream x and the arena.
type scratch struct {
	x  []float32
	ar arena
}

var scratchPool = sync.Pool{New: func() any { return new(scratch) }}
