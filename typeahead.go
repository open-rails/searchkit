package searchkit

import (
	"context"
	"sort"
	"strings"

	"github.com/doujins-org/searchkit/search"
	"github.com/jackc/pgx/v5/pgxpool"
)

func isCJKLanguage(lang string) bool {
	switch strings.ToLower(strings.TrimSpace(lang)) {
	case "ja", "zh", "ko":
		return true
	default:
		return false
	}
}

// Typeahead is the recommended entrypoint for trigram-based suggestions while typing.
//
// Under the hood it uses:
//   - `pg_trgm` over `<schema>.search_documents.document` for most languages
//   - PGroonga over `<schema>.search_documents.raw_document` for ja/zh/ko (native script)
func Typeahead(ctx context.Context, pool *pgxpool.Pool, query string, opts search.LexicalOptions) ([]search.LexicalHit, error) {
	q := normalizeWhitespace(query)
	if q == "" || !hasAnyLetterOrNumber(q) {
		return []search.LexicalHit{}, nil
	}

	if !isCJKLanguage(opts.Language) {
		return search.LexicalSearch(ctx, pool, q, opts)
	}

	// For ja/zh/ko:
	// - If the user types native script, use PGroonga.
	// - If the user types ASCII (romaji/pinyin), fall back to trigram.
	// - If mixed, run both and merge (max score per entity).
	usePGroonga := containsCJKScript(q)
	useTrigram := containsASCIIAlphaNum(q)

	type key struct {
		t string
		i string
		l string
	}
	merged := make(map[key]search.LexicalHit)

	add := func(h search.LexicalHit) {
		k := key{t: h.EntityType, i: h.EntityID, l: h.Language}
		if prev, ok := merged[k]; !ok || h.Score > prev.Score {
			merged[k] = h
		}
	}

	if useTrigram {
		hits, err := search.LexicalSearch(ctx, pool, q, opts)
		if err != nil {
			return nil, err
		}
		for _, h := range hits {
			add(h)
		}
	}

	if usePGroonga {
		hits, err := search.PGroongaSearch(ctx, pool, q, search.PGroongaOptions{
			Schema:      opts.Schema,
			Language:    opts.Language,
			EntityTypes: opts.EntityTypes,
			Limit:       opts.Limit,
			Prefix:      true,
			ScoreK:      1,
		})
		if err != nil {
			return nil, err
		}
		for _, h := range hits {
			if opts.MinSimilarity > 0 && h.Score < opts.MinSimilarity {
				continue
			}
			add(search.LexicalHit{
				EntityType: h.EntityType,
				EntityID:   h.EntityID,
				Language:   h.Language,
				Score:      h.Score,
			})
		}
	}

	out := make([]search.LexicalHit, 0, len(merged))
	for _, h := range merged {
		out = append(out, h)
	}

	// Re-sort deterministically by score desc then key asc, and apply limit.
	// (Trigram and PGroonga may return different internal orders.)
	sortByScoreThenKey(out)
	if opts.Limit > 0 && len(out) > opts.Limit {
		out = out[:opts.Limit]
	}
	return out, nil
}

func sortByScoreThenKey(hits []search.LexicalHit) {
	sort.Slice(hits, func(i, j int) bool {
		a, b := hits[i], hits[j]
		if a.Score != b.Score {
			return a.Score > b.Score
		}
		if a.EntityType != b.EntityType {
			return a.EntityType < b.EntityType
		}
		if a.EntityID != b.EntityID {
			return a.EntityID < b.EntityID
		}
		return a.Language < b.Language
	})
}
