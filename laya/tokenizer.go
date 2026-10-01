// Package laya runs the Laya multilingual "System One" classifier locally, in Go:
// HF-compatible tokenizer, mmBERT encoder (GGUF via llama.cpp), and the decision head.
package laya

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"unicode/utf8"
)

// Tokenizer is a Go port of the HF `tokenizers` pipeline used by Laya's tokenizer.json files.
// Two layouts are supported, picked from the file:
//
//   - Gemma-style (mmBERT): added-token splitting -> Replace(" ","▁") -> Metaspace(prepend always,
//     split) -> BPE with byte fallback.
//   - GPT-2-style (ModernBERT): added-token splitting -> NFC -> ByteLevel(regex split, no prefix
//     space) -> BPE over the byte-to-unicode alphabet.
//
// ponytail: only these two layouts, not the general HF format.
type Tokenizer struct {
	vocab    map[string]int32
	ranks    map[[2]string]int32
	unk      int32
	added    []addedToken
	trieRoot *trieNode
	mu       sync.Mutex         // guards cache
	cache    map[string][]int32 // BPE result per pre-token

	byteLevel bool // GPT-2-style ByteLevel pre-tokenizer instead of Metaspace

	// Special token ids used to build sequences.
	CLS, SEP, Mask, Pad int32
	// MaskText is the mask token's text. It must not appear in user text (the tokenizer would
	// turn it into the mask token), so callers replace it with a space first.
	MaskText string
}

type addedToken struct {
	id      int32
	content string
	lstrip  bool
}

type trieNode struct {
	next map[byte]*trieNode
	tok  int // index into Tokenizer.added, -1 when no token ends here
}

const metaspace = "▁" // ▁

// LoadTokenizer reads tokenizer.json.
func LoadTokenizer(path string) (*Tokenizer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("laya: read tokenizer: %w", err)
	}

	var f struct {
		AddedTokens []struct {
			ID      int32  `json:"id"`
			Content string `json:"content"`
			LStrip  bool   `json:"lstrip"`
		} `json:"added_tokens"`
		Model struct {
			Type         string            `json:"type"`
			Vocab        map[string]int32  `json:"vocab"`
			Merges       []json.RawMessage `json:"merges"`
			Unk          *string           `json:"unk_token"`
			ByteFallback bool              `json:"byte_fallback"`
			IgnoreMerges bool              `json:"ignore_merges"`
			Dropout      *float64          `json:"dropout"`
		} `json:"model"`
		Normalizer struct {
			Type    string `json:"type"`
			Content string `json:"content"`
		} `json:"normalizer"`
		PreTokenizer struct {
			Type           string `json:"type"`
			Replacement    string `json:"replacement"`
			PrependScheme  string `json:"prepend_scheme"`
			Split          bool   `json:"split"`
			AddPrefixSpace bool   `json:"add_prefix_space"`
			UseRegex       bool   `json:"use_regex"`
		} `json:"pre_tokenizer"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, fmt.Errorf("laya: parse tokenizer: %w", err)
	}

	byteLevel := f.PreTokenizer.Type == "ByteLevel"
	switch {
	case byteLevel:
		if f.Model.Type != "BPE" || f.Model.ByteFallback || f.Model.IgnoreMerges || f.Model.Dropout != nil {
			return nil, fmt.Errorf("laya: unsupported tokenizer model (need plain BPE without byte_fallback)")
		}
		if f.Normalizer.Type != "" && f.Normalizer.Type != "NFC" {
			return nil, fmt.Errorf("laya: unsupported normalizer %q (need NFC or none)", f.Normalizer.Type)
		}
		if f.PreTokenizer.AddPrefixSpace || !f.PreTokenizer.UseRegex {
			return nil, fmt.Errorf("laya: unsupported ByteLevel pre_tokenizer (need use_regex, no prefix space)")
		}
	case f.Model.Type != "BPE" || !f.Model.ByteFallback || f.Model.IgnoreMerges || f.Model.Dropout != nil:
		return nil, fmt.Errorf("laya: unsupported tokenizer model (need plain BPE with byte_fallback)")
	case f.Normalizer.Type != "Replace" || f.Normalizer.Content != metaspace:
		return nil, fmt.Errorf("laya: unsupported normalizer %q", f.Normalizer.Type)
	case f.PreTokenizer.Type != "Metaspace" || f.PreTokenizer.Replacement != metaspace ||
		f.PreTokenizer.PrependScheme != "always" || !f.PreTokenizer.Split:
		return nil, fmt.Errorf("laya: unsupported pre_tokenizer %q", f.PreTokenizer.Type)
	}

	t := &Tokenizer{
		vocab:     f.Model.Vocab,
		ranks:     make(map[[2]string]int32, len(f.Model.Merges)),
		cache:     map[string][]int32{},
		byteLevel: byteLevel,
	}
	for i, m := range f.Model.Merges {
		var pair [2]string
		if err := json.Unmarshal(m, &pair); err != nil {
			// older format: "a b"
			var s string
			if err2 := json.Unmarshal(m, &s); err2 != nil {
				return nil, fmt.Errorf("laya: bad merge %d", i)
			}
			a, b, ok := strings.Cut(s, " ")
			if !ok {
				return nil, fmt.Errorf("laya: bad merge %q", s)
			}
			pair = [2]string{a, b}
		}
		t.ranks[pair] = int32(i)
	}

	t.unk = -1 // ByteLevel BPE has no unk: every byte is in the alphabet
	if f.Model.Unk != nil {
		var ok bool
		if t.unk, ok = t.vocab[*f.Model.Unk]; !ok {
			return nil, fmt.Errorf("laya: unk token %q not in vocab", *f.Model.Unk)
		}
	} else if !byteLevel {
		return nil, fmt.Errorf("laya: tokenizer has no unk_token")
	}

	t.trieRoot = &trieNode{next: map[byte]*trieNode{}, tok: -1}
	for _, a := range f.AddedTokens {
		t.added = append(t.added, addedToken{id: a.ID, content: a.Content, lstrip: a.LStrip})
		n := t.trieRoot
		for i := 0; i < len(a.Content); i++ {
			c := a.Content[i]
			if n.next[c] == nil {
				n.next[c] = &trieNode{next: map[byte]*trieNode{}, tok: -1}
			}
			n = n.next[c]
		}
		n.tok = len(t.added) - 1
	}

	names := map[string]*int32{"<bos>": &t.CLS, "<eos>": &t.SEP, "<mask>": &t.Mask, "<pad>": &t.Pad}
	t.MaskText = "<mask>"
	if byteLevel {
		names = map[string]*int32{"[CLS]": &t.CLS, "[SEP]": &t.SEP, "[MASK]": &t.Mask, "[PAD]": &t.Pad}
		t.MaskText = "[MASK]"
	}
	for name, dst := range names {
		id, ok := t.vocab[name]
		if !ok { // ModernBERT's special tokens live in added_tokens, not the vocab
			for _, a := range t.added {
				if a.content == name {
					id, ok = a.id, true
				}
			}
		}
		if !ok {
			return nil, fmt.Errorf("laya: special token %s missing", name)
		}
		*dst = id
	}
	return t, nil
}

// Encode tokenizes text without adding special tokens (HF add_special_tokens=False).
func (t *Tokenizer) Encode(text string) []int32 {
	t.mu.Lock()
	defer t.mu.Unlock()

	var ids []int32
	chunk := 0 // start of the pending plain-text chunk

	flush := func(end int) {
		if end > chunk {
			ids = append(ids, t.encodePlain(text[chunk:end])...)
		}
	}

	for i := 0; i < len(text); {
		tok := t.longestAdded(text[i:])
		if tok < 0 {
			i++
			continue
		}
		a := t.added[tok]
		end := i
		if a.lstrip { // the token swallows whitespace to its left
			for end > chunk {
				r, size := utf8.DecodeLastRuneInString(text[chunk:end])
				if !isSpace(r) {
					break
				}
				end -= size
			}
		}
		flush(end)
		ids = append(ids, a.id)
		i += len(a.content)
		chunk = i
	}
	flush(len(text))
	return ids
}

// longestAdded returns the index of the longest added token that prefixes s, or -1.
func (t *Tokenizer) longestAdded(s string) int {
	n, best := t.trieRoot, -1
	for i := 0; i < len(s); i++ {
		if n = n.next[s[i]]; n == nil {
			break
		}
		if n.tok >= 0 {
			best = n.tok
		}
	}
	return best
}

func isSpace(r rune) bool {
	switch r {
	case ' ', '\t', '\n', '\v', '\f', '\r', 0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000:
		return true
	}
	return r >= 0x2000 && r <= 0x200A
}

// encodePlain runs normalizer, pre-tokenizer and BPE on text without added tokens.
func (t *Tokenizer) encodePlain(text string) []int32 {
	if t.byteLevel {
		return t.encodeByteLevel(text)
	}
	text = strings.ReplaceAll(text, " ", metaspace)
	if !strings.HasPrefix(text, metaspace) { // prepend_scheme=always
		text = metaspace + text
	}

	var ids []int32
	// Metaspace split, MergedWithNext: every ▁ starts a new piece.
	start := 0
	for i := 0; i < len(text); {
		if strings.HasPrefix(text[i:], metaspace) && i > start {
			ids = append(ids, t.bpe(text[start:i])...)
			start = i
		}
		_, size := utf8.DecodeRuneInString(text[i:])
		i += size
	}
	if start < len(text) {
		ids = append(ids, t.bpe(text[start:])...)
	}
	return ids
}

// bpe encodes one pre-token: chars (with byte fallback), then greedy lowest-rank merges.
func (t *Tokenizer) bpe(piece string) []int32 {
	if r, ok := t.cache[piece]; ok {
		return r
	}

	var syms []string
	for _, r := range piece {
		s := string(r)
		if _, ok := t.vocab[s]; ok {
			syms = append(syms, s)
			continue
		}
		// byte fallback: <0xXX> per UTF-8 byte; if any is missing the char is <unk>
		var bytes []string
		for _, b := range []byte(s) {
			name := fmt.Sprintf("<0x%02X>", b)
			if _, ok := t.vocab[name]; !ok {
				bytes = nil
				break
			}
			bytes = append(bytes, name)
		}
		if bytes == nil {
			syms = append(syms, "\x00unk")
		} else {
			syms = append(syms, bytes...)
		}
	}

	for len(syms) > 1 {
		best, bestRank := -1, int32(-1)
		for i := 0; i+1 < len(syms); i++ {
			if r, ok := t.ranks[[2]string{syms[i], syms[i+1]}]; ok && (bestRank < 0 || r < bestRank) {
				best, bestRank = i, r
			}
		}
		if best < 0 {
			break
		}
		syms[best] += syms[best+1]
		syms = append(syms[:best+1], syms[best+2:]...)
	}

	ids := make([]int32, 0, len(syms))
	for _, s := range syms {
		if s == "\x00unk" {
			if n := len(ids); n > 0 && ids[n-1] == t.unk { // fuse_unk
				continue
			}
			ids = append(ids, t.unk)
			continue
		}
		ids = append(ids, t.vocab[s])
	}
	t.cache[piece] = ids
	return ids
}
