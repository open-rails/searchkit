package search

import (
	"context"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/doujins-org/searchkit/internal/textnormalize"
)

type LexicalHit struct {
	EntityType string
	EntityID   string
	Language   string
	Score      float32
}

type LexicalOptions struct {
	Schema        string
	Language      string
	EntityTypes   []string
	Limit         int
	MinSimilarity float32
}

// LexicalSearch runs a trigram similarity search against `<schema>.search_documents`.
//
// searchkit heavy-normalizes the query (and expects stored documents to be heavy-normalized
// at write time).
func LexicalSearch(ctx context.Context, pool *pgxpool.Pool, query string, opts LexicalOptions) ([]LexicalHit, error) {
	if pool == nil {
		return nil, fmt.Errorf("pool is required")
	}
	if strings.TrimSpace(opts.Schema) == "" {
		return nil, fmt.Errorf("schema is required")
	}
	if strings.TrimSpace(opts.Language) == "" {
		return nil, fmt.Errorf("language is required")
	}
	if opts.Limit <= 0 {
		return []LexicalHit{}, nil
	}

	q := textnormalize.Heavy(query)
	if q == "" {
		return []LexicalHit{}, nil
	}

	quotedSchema, err := quoteIdent(opts.Schema)
	if err != nil {
		return nil, fmt.Errorf("invalid schema: %w", err)
	}
	table := quotedSchema + ".search_documents"

	where := "WHERE sd.language = @language"
	args := pgx.NamedArgs{
		"language": opts.Language,
		"q":        q,
		"limit":    opts.Limit,
	}
	if len(opts.EntityTypes) > 0 {
		where += " AND sd.entity_type = ANY(@entity_types::text[])"
		args["entity_types"] = opts.EntityTypes
	}

	// Use both `%` (fast candidate filter via gin_trgm_ops) and similarity threshold.
	// Note: `%` is sensitive to pg_trgm similarity threshold setting; we still apply
	// an explicit SIMILARITY(...) >= minSimilarity filter.
	minSim := opts.MinSimilarity
	if minSim <= 0 {
		minSim = 0.1
	}
	args["min_similarity"] = minSim

	sql := fmt.Sprintf(`
		SELECT
			sd.entity_type,
			sd.entity_id,
			sd.language,
			SIMILARITY(sd.document, @q)::float4 AS score
		FROM %s sd
		%s
		  AND sd.document %% @q
		  AND SIMILARITY(sd.document, @q) >= @min_similarity
		ORDER BY score DESC, sd.entity_type ASC, sd.entity_id ASC
		LIMIT @limit
	`, table, where)

	rows, err := pool.Query(ctx, sql, args)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []LexicalHit
	for rows.Next() {
		var h LexicalHit
		if err := rows.Scan(&h.EntityType, &h.EntityID, &h.Language, &h.Score); err != nil {
			return nil, err
		}
		out = append(out, h)
	}
	return out, rows.Err()
}
