package textnormalize

import (
	"strings"
	"unicode"

	"github.com/mozillazg/go-unidecode"
	"golang.org/x/text/unicode/norm"
)

// Heavy normalizes text for lexical trigram search:
// - Unicode NFKC
// - transliteration to ASCII (best-effort)
// - lowercase
// - punctuation collapse to spaces
// - whitespace collapse
//
// It is intentionally language-agnostic and conservative: it aims to make
// cross-script matching possible (e.g. 日本語 vs romaji) while staying stable.
func Heavy(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}

	s = norm.NFKC.String(s)
	s = unidecode.Unidecode(s)
	s = strings.ToLower(s)

	var b strings.Builder
	b.Grow(len(s))

	space := false
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			if space && b.Len() > 0 {
				b.WriteByte(' ')
			}
			space = false
			b.WriteRune(r)
			continue
		}
		space = true
	}

	out := strings.TrimSpace(b.String())
	if out == "" {
		return ""
	}
	// Final collapse in case of leading/trailing spaces.
	return strings.Join(strings.Fields(out), " ")
}
