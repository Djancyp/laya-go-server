package laya

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"regexp"
	"strings"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// Unigram is a Go port of the HF `tokenizers` pipeline of DeBERTa-v3 (SentencePiece Unigram):
// added-token splitting -> normalizer (collapse whitespace, NFC, right strip) -> Metaspace
// (prepend always, split) -> Viterbi over the piece scores.
//
// ponytail: only this layout, not the general HF format.
type Unigram struct {
	pieces  map[string]unigramPiece
	maxLen  int // longest piece, in runes
	unk     int32
	unkCost float64
	added   []addedToken
	trie    *trieNode
}

type unigramPiece struct {
	id    int32
	score float64
}

var wsRun = regexp.MustCompile(`\s{2,}|[\n\r\t]`)

// LoadUnigram reads a Unigram tokenizer.json.
func LoadUnigram(path string) (*Unigram, error) {
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
			Type         string   `json:"type"`
			Vocab        [][2]any `json:"vocab"`
			Unk          *int32   `json:"unk_id"`
			ByteFallback bool     `json:"byte_fallback"`
		} `json:"model"`
		PreTokenizer struct {
			Type          string `json:"type"`
			Replacement   string `json:"replacement"`
			PrependScheme string `json:"prepend_scheme"`
			Split         bool   `json:"split"`
			PreTokenizers []struct {
				Type          string `json:"type"`
				Replacement   string `json:"replacement"`
				PrependScheme string `json:"prepend_scheme"`
				Split         bool   `json:"split"`
			} `json:"pretokenizers"`
		} `json:"pre_tokenizer"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, fmt.Errorf("laya: parse tokenizer: %w", err)
	}
	if f.Model.Type != "Unigram" || f.Model.ByteFallback || f.Model.Unk == nil {
		return nil, fmt.Errorf("laya: unsupported tokenizer model %q (need Unigram with unk_id, no byte_fallback)", f.Model.Type)
	}
	// The pre_tokenizer is a Metaspace, alone or as the only member of a Sequence.
	p := f.PreTokenizer
	if p.Type == "Sequence" && len(p.PreTokenizers) == 1 {
		p.Type, p.Replacement, p.PrependScheme, p.Split = p.PreTokenizers[0].Type, p.PreTokenizers[0].Replacement, p.PreTokenizers[0].PrependScheme, p.PreTokenizers[0].Split
	}
	if p.Type != "Metaspace" || p.Replacement != metaspace || p.PrependScheme != "always" || !p.Split {
		return nil, fmt.Errorf("laya: unsupported pre_tokenizer %q (need Metaspace, prepend always, split)", p.Type)
	}

	u := &Unigram{pieces: make(map[string]unigramPiece, len(f.Model.Vocab)), unk: *f.Model.Unk}
	minScore := math.Inf(1)
	for i, e := range f.Model.Vocab {
		s, ok1 := e[0].(string)
		sc, ok2 := e[1].(float64)
		if !ok1 || !ok2 {
			return nil, fmt.Errorf("laya: bad vocab entry %d", i)
		}
		u.pieces[s] = unigramPiece{id: int32(i), score: sc}
		u.maxLen = max(u.maxLen, utf8.RuneCountInString(s))
		minScore = math.Min(minScore, sc)
	}
	u.unkCost = minScore - 10 // tokenizers' K_UNK_PENALTY

	u.trie = &trieNode{next: map[byte]*trieNode{}, tok: -1}
	for _, a := range f.AddedTokens {
		u.added = append(u.added, addedToken{id: a.ID, content: a.Content, lstrip: a.LStrip})
		n := u.trie
		for i := 0; i < len(a.Content); i++ {
			c := a.Content[i]
			if n.next[c] == nil {
				n.next[c] = &trieNode{next: map[byte]*trieNode{}, tok: -1}
			}
			n = n.next[c]
		}
		n.tok = len(u.added) - 1
	}
	return u, nil
}

// TokenID is the id of an added token such as "[L]".
func (u *Unigram) TokenID(content string) (int32, bool) {
	for _, a := range u.added {
		if a.content == content {
			return a.id, true
		}
	}
	return 0, false
}

// Encode tokenizes text without adding special tokens (HF tokenize()), so added tokens such as
// "[P]" inside text become their single id.
func (u *Unigram) Encode(text string) []int32 {
	var ids []int32
	chunk := 0
	for i := 0; i < len(text); {
		tok := longestAddedIn(u.trie, text[i:])
		if tok < 0 {
			i++
			continue
		}
		if i > chunk {
			ids = append(ids, u.encodePlain(text[chunk:i])...)
		}
		ids = append(ids, u.added[tok].id)
		i += len(u.added[tok].content)
		chunk = i
	}
	if len(text) > chunk {
		ids = append(ids, u.encodePlain(text[chunk:])...)
	}
	return ids
}

func longestAddedIn(root *trieNode, s string) int {
	n, best := root, -1
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

func (u *Unigram) encodePlain(text string) []int32 {
	text = wsRun.ReplaceAllString(text, " ")
	text = strings.TrimRight(norm.NFC.String(text), " ")
	if text == "" {
		return nil
	}
	text = strings.ReplaceAll(text, " ", metaspace)
	if !strings.HasPrefix(text, metaspace) {
		text = metaspace + text
	}
	var ids []int32
	// Metaspace split, MergedWithNext: every ▁ starts a new piece.
	start := 0
	for i := 0; i < len(text); {
		if strings.HasPrefix(text[i:], metaspace) && i > start {
			ids = u.viterbi(ids, text[start:i])
			start = i
		}
		_, size := utf8.DecodeRuneInString(text[i:])
		i += size
	}
	return u.viterbi(ids, text[start:])
}

// viterbi appends the best segmentation of s. Characters no piece covers become the unk id,
// consecutive ones fused into one (tokenizers' fuse_unk).
func (u *Unigram) viterbi(out []int32, s string) []int32 {
	// byte offsets of every rune boundary
	offs := make([]int, 0, len(s)+1)
	for i := range s {
		offs = append(offs, i)
	}
	n := len(offs)
	offs = append(offs, len(s))

	type node struct {
		score float64
		from  int // rune index the best last piece starts at
		id    int32
		unk   bool
	}
	best := make([]node, n+1)
	for i := 1; i <= n; i++ {
		best[i].score = math.Inf(-1)
	}
	for i := range n {
		if math.IsInf(best[i].score, -1) {
			continue
		}
		found := false
		for j := i + 1; j <= min(n, i+u.maxLen); j++ {
			p, ok := u.pieces[s[offs[i]:offs[j]]]
			if !ok {
				continue
			}
			if j == i+1 {
				found = true
			}
			if sc := best[i].score + p.score; sc > best[j].score {
				best[j] = node{score: sc, from: i, id: p.id}
			}
		}
		if !found {
			if sc := best[i].score + u.unkCost; sc > best[i+1].score {
				best[i+1] = node{score: sc, from: i, id: u.unk, unk: true}
			}
		}
	}

	var rev []int32
	for j := n; j > 0; j = best[j].from {
		if best[j].unk && len(rev) > 0 && rev[len(rev)-1] == u.unk {
			continue // fuse_unk
		}
		rev = append(rev, best[j].id)
	}
	for i := len(rev) - 1; i >= 0; i-- {
		out = append(out, rev[i])
	}
	return out
}
