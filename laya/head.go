package laya

import (
	"encoding/json"
	"fmt"
	"math"
	"runtime"
	"sync"
)

// Head is the Laya decision head: type embedding, a stack of pre-norm transformer
// encoder layers (PyTorch nn.TransformerEncoderLayer, relu, norm_first) and an MLP
// scorer read at each option's [MASK] marker.
// ponytail: the "act"/escalate head is not ported (Predict does not expose it).
type Head struct {
	d, heads, ff int
	typeEmb      []float32 // 3 x d
	layers       []headLayer
	scorer       scorer
}

type headLayer struct {
	inW, inB   []float32 // 3d x d, 3d   (q, k, v stacked)
	outW, outB []float32 // d x d, d
	ln1W, ln1B []float32
	ln2W, ln2B []float32
	l1W, l1B   []float32 // ff x d, ff
	l2W, l2B   []float32 // d x ff, d
}

type scorer struct {
	lnW, lnB []float32
	w1, b1   []float32 // d x d
	w2, b2   []float32 // 1 x d
}

// HeadConfig is the "laya.config" metadata stored in the head file.
type HeadConfig struct {
	Encoder     string     `json:"encoder"`
	HeadLayers  int        `json:"head_layers"`
	MaxLen      int        `json:"max_len"`
	HeadMaxLen  int        `json:"head_max_len"`
	Temperature [3]float64 `json:"temperature"`
	// TemperatureByOptions is "type:size" -> temperature, e.g. "choice:6-10".
	TemperatureByOptions map[string]float64 `json:"temperature_by_options"`
}

const (
	hiddenDim = 768 // mmBERT-base; only the tests' default, the head reads its width from type_emb.weight
	headDim   = 64  // PyTorch nhead = d/64
	lnEps     = 1e-5
)

// LoadHead reads laya-multilingual-head.safetensors.
func LoadHead(path string) (*Head, *HeadConfig, error) {
	t, err := LoadSafetensors(path)
	if err != nil {
		return nil, nil, err
	}

	var cfg HeadConfig
	if err := json.Unmarshal([]byte(t.Meta["laya.config"]), &cfg); err != nil {
		return nil, nil, fmt.Errorf("laya: head config: %w", err)
	}
	if cfg.HeadLayers < 0 || cfg.HeadLayers > 8 {
		return nil, nil, fmt.Errorf("laya: unsupported head_layers=%d", cfg.HeadLayers)
	}

	// The width comes from the file (768 for mmBERT-base, 1024 for ModernBERT-large).
	shape, ok := t.Shape("type_emb.weight")
	if !ok || len(shape) != 2 || shape[0] != 3 || shape[1] < headDim || shape[1]%headDim != 0 {
		return nil, nil, fmt.Errorf("laya: type_emb.weight has shape %v, want [3 d] with d a multiple of %d", shape, headDim)
	}
	d := shape[1]
	h := &Head{d: d, heads: d / headDim, ff: 4 * d}

	var firstErr error
	get := func(name string, shape ...int) []float32 {
		v, err := t.Get(name, shape...)
		if err != nil && firstErr == nil {
			firstErr = err
		}
		return v
	}

	h.typeEmb = get("type_emb.weight", 3, d)
	for i := 0; i < cfg.HeadLayers; i++ {
		p := fmt.Sprintf("head.layers.%d.", i)
		h.layers = append(h.layers, headLayer{
			inW: get(p+"self_attn.in_proj_weight", 3*d, d), inB: get(p+"self_attn.in_proj_bias", 3*d),
			outW: get(p+"self_attn.out_proj.weight", d, d), outB: get(p+"self_attn.out_proj.bias", d),
			ln1W: get(p+"norm1.weight", d), ln1B: get(p+"norm1.bias", d),
			ln2W: get(p+"norm2.weight", d), ln2B: get(p+"norm2.bias", d),
			l1W: get(p+"linear1.weight", h.ff, d), l1B: get(p+"linear1.bias", h.ff),
			l2W: get(p+"linear2.weight", d, h.ff), l2B: get(p+"linear2.bias", d),
		})
	}
	h.scorer = scorer{
		lnW: get("scorer.0.weight", d), lnB: get("scorer.0.bias", d),
		w1: get("scorer.1.weight", d, d), b1: get("scorer.1.bias", d),
		w2: get("scorer.3.weight", 1, d), b2: get("scorer.3.bias", 1),
	}
	if firstErr != nil {
		return nil, nil, firstErr
	}
	return h, &cfg, nil
}

// Dim is the encoder hidden size the head expects.
func (h *Head) Dim() int { return h.d }

// Logits scores every option marker. hidden is the encoder's last_hidden_state for one
// sequence (n x Dim(), row-major); qtype is 0 choice / 1 score / 2 noul; markers are the
// token positions of each option's [MASK]. It returns one logit per marker.
func (h *Head) Logits(hidden []float32, qtype int, markers []int) ([]float32, error) {
	d := h.d
	if len(hidden) == 0 || len(hidden)%d != 0 {
		return nil, fmt.Errorf("laya: hidden has %d values, not a multiple of %d", len(hidden), d)
	}
	n := len(hidden) / d
	if qtype < 0 || qtype > 2 {
		return nil, fmt.Errorf("laya: bad question type %d", qtype)
	}
	for _, m := range markers {
		if m < 0 || m >= n {
			return nil, fmt.Errorf("laya: marker %d outside sequence of %d tokens", m, n)
		}
	}

	sc, ok := scratchPool.Get().(*scratch)
	if !ok {
		sc = new(scratch)
	}
	defer scratchPool.Put(sc)
	ar := &sc.ar
	if cap(sc.x) < len(hidden) {
		sc.x = make([]float32, len(hidden))
	}
	x := sc.x[:len(hidden)] // the residual stream, updated in place by each layer
	emb := h.typeEmb[qtype*d : (qtype+1)*d]
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			x[i*d+j] = hidden[i*d+j] + emb[j]
		}
	}

	rows := n // number of rows in x after each layer
	for li, l := range h.layers {
		if li == len(h.layers)-1 {
			// Only the marker rows are read afterwards, so the last layer computes queries,
			// projection and feed-forward for those rows alone (K and V still need every token).
			x = h.layer(ar, l, x, n, markers)
			rows = len(markers)
		} else {
			x = h.layer(ar, l, x, n, nil)
		}
	}
	if len(h.layers) == 0 {
		rows = len(markers)
		x = pick(nil, x, markers, d)
	}
	ar.reset() // x is never arena memory here: sc.x, or a fresh pick

	// scorer: LayerNorm -> Linear -> GELU -> Linear(d,1)
	s := h.scorer
	z := layerNorm(ar, x, rows, d, s.lnW, s.lnB)
	z = linear(ar, z, rows, d, s.w1, s.b1, d)
	for i, v := range z {
		z[i] = float32(0.5 * float64(v) * (1 + math.Erf(float64(v)/math.Sqrt2)))
	}
	out := linear(nil, z, rows, d, s.w2, s.b2, 1) // returned: must not be pooled memory
	return out, nil
}

// layer applies one pre-norm encoder layer. x is n x d. When qRows is nil, all n rows are
// computed and x itself is updated in place and returned; otherwise only those rows are
// computed, into a fresh len(qRows) x d slice. Temporaries come from ar, reset per phase.
func (h *Head) layer(ar *arena, l headLayer, x []float32, n int, qRows []int) []float32 {
	d := h.d

	// attention block: a, k, v, q, ctx and o are dead once o is added to the residual
	ar.reset()
	a := layerNorm(ar, x, n, d, l.ln1W, l.ln1B)
	k := linear(ar, a, n, d, l.inW[d*d:2*d*d], l.inB[d:2*d], d)
	v := linear(ar, a, n, d, l.inW[2*d*d:], l.inB[2*d:], d)

	// query rows and the residual stream for them (outside the arena: it survives resets)
	nq := n
	qa, y := a, x
	if qRows != nil {
		nq = len(qRows)
		qa, y = pick(ar, a, qRows, d), pick(nil, x, qRows, d)
	}
	q := linear(ar, qa, nq, d, l.inW[:d*d], l.inB[:d], d)

	ctx := h.attention(ar, q, k, v, nq, n)
	o := linear(ar, ctx, nq, d, l.outW, l.outB, d)
	for i := range y {
		y[i] += o[i]
	}

	// feed-forward block
	ar.reset()
	b := layerNorm(ar, y, nq, d, l.ln2W, l.ln2B)
	f := linear(ar, b, nq, d, l.l1W, l.l1B, h.ff)
	for i, val := range f {
		if val < 0 {
			f[i] = 0
		}
	}
	g := linear(ar, f, nq, h.ff, l.l2W, l.l2B, d)
	for i := range y {
		y[i] += g[i]
	}
	return y
}

func pick(ar *arena, x []float32, rows []int, d int) []float32 {
	out := ar.get(len(rows) * d)
	for i, r := range rows {
		copy(out[i*d:(i+1)*d], x[r*d:(r+1)*d])
	}
	return out
}

// attention computes softmax(q k^T / sqrt(64)) v per head (no mask: sequences are unpadded).
// q is nq x d, k and v are nk x d; the result is nq x d.
func (h *Head) attention(ar *arena, q, k, v []float32, nq, nk int) []float32 {
	d := h.d
	out := ar.get(nq * d)
	clear(out) // accumulated into with axpy
	scale := float32(1 / math.Sqrt(headDim))

	parallel(h.heads, func(head int) {
		off := head * headDim
		scores := make([]float32, nk)
		for i := 0; i < nq; i++ {
			qi := q[i*d+off : i*d+off+headDim]
			maxv := float32(math.Inf(-1))
			for j := 0; j < nk; j++ {
				s := dot(qi, k[j*d+off:j*d+off+headDim]) * scale
				scores[j] = s
				if s > maxv {
					maxv = s
				}
			}
			var sum float32
			for j := range scores {
				scores[j] = float32(math.Exp(float64(scores[j] - maxv)))
				sum += scores[j]
			}
			dst := out[i*d+off : i*d+off+headDim]
			for j := 0; j < nk; j++ {
				w := scores[j] / sum
				axpy(dst, v[j*d+off:j*d+off+headDim], w)
			}
		}
	})
	return out
}

// layerNorm normalizes each of the rows over its d values (biased variance, eps 1e-5).
func layerNorm(ar *arena, x []float32, rows, d int, w, b []float32) []float32 {
	out := ar.get(rows * d)
	for i := 0; i < rows; i++ {
		row := x[i*d : (i+1)*d]
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(d)
		var variance float64
		for _, v := range row {
			dv := float64(v) - mean
			variance += dv * dv
		}
		inv := 1 / math.Sqrt(variance/float64(d)+lnEps)
		for j, v := range row {
			out[i*d+j] = float32((float64(v)-mean)*inv)*w[j] + b[j]
		}
	}
	return out
}

// linear computes y = x W^T + b for rows x in-dim input and a (outDim x in) weight.
// Work is split across output tiles so each tile of W stays cache-resident across rows.
func linear(ar *arena, x []float32, rows, in int, w, b []float32, outDim int) []float32 {
	y := ar.get(rows * outDim) // every element is written below
	const tile = 32
	tiles := (outDim + tile - 1) / tile

	parallel(tiles, func(t int) {
		lo, hi := t*tile, min((t+1)*tile, outDim)
		for i := 0; i < rows; i++ {
			xi := x[i*in : (i+1)*in]
			for o := lo; o < hi; o++ {
				y[i*outDim+o] = dot(xi, w[o*in:(o+1)*in]) + b[o]
			}
		}
	})
	return y
}

// dotScalar is a 4-way unrolled float32 dot product: the portable path behind dot
// (dot_scalar.go, or dot_simd.go on CPUs without AVX2+FMA).
func dotScalar(a, b []float32) float32 {
	var s0, s1, s2, s3 float32
	n := len(a)
	b = b[:n]
	i := 0
	for ; i+4 <= n; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < n; i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

// axpyScalar is dst += alpha * x, the portable path behind axpy.
func axpyScalar(dst, x []float32, alpha float32) {
	x = x[:len(dst)]
	for i := range dst {
		dst[i] += alpha * x[i]
	}
}

// parallel runs fn(0..n-1) across up to NumCPU goroutines.
func parallel(n int, fn func(i int)) {
	workers := min(runtime.NumCPU(), n)
	if workers <= 1 {
		for i := 0; i < n; i++ {
			fn(i)
		}
		return
	}
	var wg sync.WaitGroup
	var mu sync.Mutex
	next := 0
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				mu.Lock()
				i := next
				next++
				mu.Unlock()
				if i >= n {
					return
				}
				fn(i)
			}
		}()
	}
	wg.Wait()
}
