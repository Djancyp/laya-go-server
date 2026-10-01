package laya

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// Tensors holds a safetensors file's tensors decoded to float32.
type Tensors struct {
	Meta map[string]string
	data map[string]tensor
}

type tensor struct {
	shape []int
	data  []float32
}

// LoadSafetensors reads a whole safetensors file (F32, F16 or BF16 tensors).
func LoadSafetensors(path string) (*Tensors, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("laya: read safetensors: %w", err)
	}
	if len(raw) < 8 {
		return nil, fmt.Errorf("laya: %s: truncated", path)
	}
	hlen := binary.LittleEndian.Uint64(raw[:8])
	if hlen > uint64(len(raw)-8) {
		return nil, fmt.Errorf("laya: %s: bad header length", path)
	}

	var header map[string]json.RawMessage
	if err := json.Unmarshal(raw[8:8+hlen], &header); err != nil {
		return nil, fmt.Errorf("laya: %s: header: %w", path, err)
	}
	body := raw[8+hlen:]

	t := &Tensors{Meta: map[string]string{}, data: map[string]tensor{}}
	for name, msg := range header {
		if name == "__metadata__" {
			if err := json.Unmarshal(msg, &t.Meta); err != nil {
				return nil, fmt.Errorf("laya: %s: metadata: %w", path, err)
			}
			continue
		}
		var h struct {
			DType   string `json:"dtype"`
			Shape   []int  `json:"shape"`
			Offsets [2]int `json:"data_offsets"`
		}
		if err := json.Unmarshal(msg, &h); err != nil {
			return nil, fmt.Errorf("laya: %s: tensor %s: %w", path, name, err)
		}
		if h.Offsets[0] < 0 || h.Offsets[1] < h.Offsets[0] || h.Offsets[1] > len(body) {
			return nil, fmt.Errorf("laya: %s: tensor %s: bad offsets", path, name)
		}
		vals, err := decode(h.DType, body[h.Offsets[0]:h.Offsets[1]])
		if err != nil {
			return nil, fmt.Errorf("laya: %s: tensor %s: %w", path, name, err)
		}
		n := 1
		for _, d := range h.Shape {
			n *= d
		}
		if n != len(vals) {
			return nil, fmt.Errorf("laya: %s: tensor %s: shape %v does not match %d values", path, name, h.Shape, len(vals))
		}
		t.data[name] = tensor{shape: h.Shape, data: vals}
	}
	return t, nil
}

// Get returns the tensor's values after checking its shape.
func (t *Tensors) Get(name string, shape ...int) ([]float32, error) {
	x, ok := t.data[name]
	if !ok {
		return nil, fmt.Errorf("laya: tensor %q missing", name)
	}
	if len(x.shape) != len(shape) {
		return nil, fmt.Errorf("laya: tensor %q has shape %v, want %v", name, x.shape, shape)
	}
	for i := range shape {
		if x.shape[i] != shape[i] {
			return nil, fmt.Errorf("laya: tensor %q has shape %v, want %v", name, x.shape, shape)
		}
	}
	return x.data, nil
}

// Shape returns a tensor's shape, or false when the tensor is absent.
func (t *Tensors) Shape(name string) ([]int, bool) {
	v, ok := t.data[name]
	return v.shape, ok
}

func decode(dtype string, b []byte) ([]float32, error) {
	switch dtype {
	case "F32":
		if len(b)%4 != 0 {
			return nil, fmt.Errorf("F32 length %d", len(b))
		}
		out := make([]float32, len(b)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
		return out, nil
	case "F16":
		if len(b)%2 != 0 {
			return nil, fmt.Errorf("F16 length %d", len(b))
		}
		out := make([]float32, len(b)/2)
		for i := range out {
			out[i] = half(binary.LittleEndian.Uint16(b[2*i:]))
		}
		return out, nil
	case "BF16":
		if len(b)%2 != 0 {
			return nil, fmt.Errorf("BF16 length %d", len(b))
		}
		out := make([]float32, len(b)/2)
		for i := range out {
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[2*i:])) << 16)
		}
		return out, nil
	}
	return nil, fmt.Errorf("unsupported dtype %s", dtype)
}

// half converts IEEE 754 binary16 to float32.
func half(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h) & 0x3ff

	switch {
	case exp == 0 && frac == 0:
		return math.Float32frombits(sign)
	case exp == 0: // subnormal: normalize
		e := uint32(127 - 15 + 1)
		for frac&0x400 == 0 {
			frac <<= 1
			e--
		}
		return math.Float32frombits(sign | e<<23 | (frac&0x3ff)<<13)
	case exp == 0x1f: // inf / nan
		return math.Float32frombits(sign | 0xff<<23 | frac<<13)
	}
	return math.Float32frombits(sign | (exp+127-15)<<23 | frac<<13)
}
