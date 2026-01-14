package searchkit

import (
	"strings"
	"unicode"
)

func isASCIIOnlyQuery(q string) bool {
	q = strings.TrimSpace(q)
	if q == "" {
		return true
	}
	for i := 0; i < len(q); i++ {
		if q[i] >= 0x80 {
			return false
		}
	}
	return true
}

func containsASCIIAlphaNum(q string) bool {
	for i := 0; i < len(q); i++ {
		c := q[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') {
			return true
		}
	}
	return false
}

func containsCJKScript(q string) bool {
	for _, r := range q {
		// CJK Symbols and Punctuation
		if r >= 0x3000 && r <= 0x303F {
			return true
		}
		// Hiragana
		if r >= 0x3040 && r <= 0x309F {
			return true
		}
		// Katakana + Katakana Phonetic Extensions
		if (r >= 0x30A0 && r <= 0x30FF) || (r >= 0x31F0 && r <= 0x31FF) {
			return true
		}
		// CJK Unified Ideographs
		if r >= 0x4E00 && r <= 0x9FFF {
			return true
		}
		// Hangul Syllables
		if r >= 0xAC00 && r <= 0xD7AF {
			return true
		}
	}
	return false
}

func normalizeWhitespace(q string) string {
	q = strings.TrimSpace(q)
	if q == "" {
		return ""
	}
	return strings.Join(strings.Fields(q), " ")
}

func hasAnyLetterOrNumber(q string) bool {
	for _, r := range q {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			return true
		}
	}
	return false
}
