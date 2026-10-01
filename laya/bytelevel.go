package laya

import (
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// byteToRune is GPT-2's bytes_to_unicode: printable bytes map to themselves, the rest to
// U+0100 and up, so every byte is one visible character of the BPE alphabet.
var byteToRune = func() (m [256]rune) {
	n := rune(0)
	for b := 0; b < 256; b++ {
		if (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF) {
			m[b] = rune(b)
		} else {
			m[b] = 256 + n
			n++
		}
	}
	return m
}()

// encodeByteLevel is NFC -> ByteLevel pre-tokenizer (GPT-2 regex split, no prefix space) -> BPE.
func (t *Tokenizer) encodeByteLevel(text string) []int32 {
	if !isASCII(text) {
		text = norm.NFC.String(text)
	}
	var ids []int32
	var sb strings.Builder
	for _, piece := range splitByteLevel(text) {
		sb.Reset()
		for i := 0; i < len(piece); i++ {
			sb.WriteRune(byteToRune[piece[i]])
		}
		ids = append(ids, t.bpe(sb.String())...)
	}
	return ids
}

func isASCII(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= utf8.RuneSelf {
			return false
		}
	}
	return true
}

// splitByteLevel splits text like the regex used by HF's ByteLevel pre-tokenizer
//
//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// Go's regexp has no lookahead, so the alternatives are tried by hand in the same order.
func splitByteLevel(s string) []string {
	var out []string
	for i := 0; i < len(s); {
		n := matchByteLevel(s[i:])
		out = append(out, s[i:i+n])
		i += n
	}
	return out
}

func runeAt(s string, i int) (rune, int) { return utf8.DecodeRuneInString(s[i:]) }

// matchByteLevel returns the length of the first match at the start of s (s is non-empty).
func matchByteLevel(s string) int {
	// 's 't 're 've 'm 'll 'd
	if s[0] == '\'' {
		for _, c := range []string{"s", "t", "re", "ve", "m", "ll", "d"} {
			if strings.HasPrefix(s[1:], c) {
				return 1 + len(c)
			}
		}
	}

	// ` ?` + one of: letters, numbers, other (not space/letter/number)
	start := 0
	if s[0] == ' ' {
		start = 1
	}
	if start < len(s) {
		r, size := runeAt(s, start)
		var in func(rune) bool
		switch {
		case unicode.IsLetter(r):
			in = unicode.IsLetter
		case unicode.IsNumber(r):
			in = unicode.IsNumber
		case !unicode.IsSpace(r):
			in = func(r rune) bool { return !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r) }
		}
		if in != nil {
			end := start + size
			for end < len(s) {
				r, size := runeAt(s, end)
				if !in(r) {
					break
				}
				end += size
			}
			return end
		}
	}

	// \s+(?!\S): the longest whitespace run that is not followed by a non-space, which leaves
	// the run's last character for the next token; \s+ takes a lone trailing-space run whole.
	end, last := 0, 0
	for end < len(s) {
		r, size := runeAt(s, end)
		if !unicode.IsSpace(r) {
			break
		}
		last = end
		end += size
	}
	if end == len(s) || last == 0 {
		return end // end of text, or a single whitespace char: \s+ (?!\S) / \s+
	}
	return last
}
