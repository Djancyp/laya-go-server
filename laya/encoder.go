package laya

import (
	"errors"
	"fmt"

	"github.com/hybridgroup/yzma/pkg/llama"
)

// Encoder runs the mmBERT encoder from a GGUF through llama.cpp and returns the
// per-token last_hidden_state (pooling disabled), which the Head consumes.
//
// llama.cpp must already be loaded and initialised (llama.Load + llama.Init; in the chat app
// kronk.Init does this), see InitLlama for standalone use.
type Encoder struct {
	model     llama.Model
	pool      chan llama.Context // idle contexts; one sequence runs per context at a time
	n         int                // pool capacity
	made      int                // contexts created so far (n unless construction failed)
	dim       int
	maxTokens int
}

// InitLlama loads the llama.cpp shared libraries from libDir unless they are already loaded.
func InitLlama(libDir string) error {
	if llama.LibPath() != "" {
		return nil
	}
	if err := llama.Load(libDir); err != nil {
		return fmt.Errorf("laya: load llama.cpp from %s: %w", libDir, err)
	}
	llama.LogSet(llama.LogSilent())
	llama.Init()
	return nil
}

// NewEncoder loads the GGUF. maxTokens bounds one sequence (the checkpoint uses max_len 1024).
// gpuLayers is the number of layers offloaded to the GPU (0 = CPU only).
func NewEncoder(path string, maxTokens, gpuLayers, contexts, threads int) (*Encoder, error) {
	if maxTokens <= 0 {
		return nil, errors.New("laya: maxTokens must be positive")
	}
	contexts = max(1, contexts)

	mp := llama.ModelDefaultParams()
	mp.NGpuLayers = int32(gpuLayers)
	model, err := llama.ModelLoadFromFile(path, mp)
	if err != nil {
		return nil, fmt.Errorf("laya: load %s: %w", path, err)
	}

	dim := int(llama.ModelNEmbd(model))
	if dim != hiddenDim {
		llama.ModelFree(model)
		return nil, fmt.Errorf("laya: model embedding size is %d, want %d", dim, hiddenDim)
	}

	// An encoder-only model needs the whole sequence in one micro-batch and every token's
	// output, so: pooling none, embeddings on, ubatch >= sequence length.
	// The contexts share the model weights; each adds only its own compute buffers.
	e := &Encoder{model: model, pool: make(chan llama.Context, contexts), n: contexts, dim: dim, maxTokens: maxTokens}
	for range contexts {
		cp := llama.ContextDefaultParams()
		cp.NCtx = uint32(maxTokens)
		cp.NBatch = uint32(maxTokens)
		cp.NUbatch = uint32(maxTokens)
		cp.NSeqMax = 1
		cp.Embeddings = 1
		cp.PoolingType = llama.PoolingTypeNone
		if threads > 0 {
			cp.NThreads, cp.NThreadsBatch = int32(threads), int32(threads)
		}
		ctx, err := llama.InitFromModel(model, cp)
		if err != nil {
			e.Close()
			return nil, fmt.Errorf("laya: create context: %w", err)
		}
		e.pool <- ctx
		e.made++
	}
	return e, nil
}

// Contexts is how many sequences can be encoded at once.
func (e *Encoder) Contexts() int { return e.n }

// Hidden returns the encoder output for ids: len(ids) x 768 floats, row-major.
func (e *Encoder) Hidden(ids []int32) ([]float32, error) {
	n := len(ids)
	if n == 0 || n > e.maxTokens {
		return nil, fmt.Errorf("laya: %d tokens, want 1..%d", n, e.maxTokens)
	}

	ctx := <-e.pool // blocks until a context is idle
	defer func() { e.pool <- ctx }()

	batch := llama.BatchInit(int32(n), 0, 1)
	defer llama.BatchFree(batch)
	for i, id := range ids {
		if err := batch.Add(llama.Token(id), llama.Pos(i), []llama.SeqId{0}, true); err != nil {
			return nil, fmt.Errorf("laya: batch: %w", err)
		}
	}

	if rc, err := llama.Decode(ctx, batch); err != nil || rc != 0 {
		return nil, fmt.Errorf("laya: encoder decode failed (rc=%d): %v", rc, err)
	}

	emb, err := llama.GetEmbeddings(ctx, n, e.dim)
	if err != nil || len(emb) != n*e.dim {
		return nil, fmt.Errorf("laya: encoder returned %d values, want %d (%v)", len(emb), n*e.dim, err)
	}

	out := make([]float32, len(emb)) // llama.cpp owns emb and reuses it on the next call
	copy(out, emb)
	return out, nil
}

// Close waits for running sequences, then releases the contexts and the model.
func (e *Encoder) Close() {
	for range e.made {
		llama.Free(<-e.pool)
	}
	e.made = 0
	if e.model != 0 {
		llama.ModelFree(e.model)
		e.model = 0
	}
}
