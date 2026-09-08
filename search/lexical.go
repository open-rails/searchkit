package search

import (
	"context"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/open-rails/searchkit/internal/textnormalize"
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

	// FilterSQL is an optional additional WHERE fragment appended to the query as:
	//   ... AND (<FilterSQL>)
	//
	// It is intended for host-owned constraints that must be enforced inside the
	// retrieval query.
	//
	// IMPORTANT: this is trusted SQL provided by the host app. Do not insert
	// user input into it unsafely.
	FilterSQL string
	// FilterArgs are named args referenced by FilterSQL using pgx '@name'
	// placeholders (e.g. "... language = @lang").
	FilterArgs map[string]any
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
	if strings.TrimSpace(opts.FilterSQL) != "" {
		where += " AND (" + opts.FilterSQL + ")"
		if err := mergeNamedArgs(args, opts.FilterArgs); err != nil {
			return nil, err
		}
	}

	minSim := opts.MinSimilarity
	if minSim <= 0 {
		minSim = 0.1
	}
	args["min_similarity"] = minSim

	// Documents concatenate every indexed field, so whole-string SIMILARITY between a
	// short query and a long document is near zero and no realistic threshold matches.
	// WORD_SIMILARITY scores the query against the best-matching extent of the document
	// instead; `<%` is its indexable form (gin_trgm_ops) and reads its threshold from
	// pg_trgm.word_similarity_threshold, set here for the statement.
	sql := fmt.Sprintf(`
		WITH _ AS (SELECT set_config('pg_trgm.word_similarity_threshold', @min_similarity::text, true))
		SELECT
			sd.entity_type,
			sd.entity_id,
			sd.language,
			WORD_SIMILARITY(@q, sd.document)::float4 AS score
		FROM _, %s sd
		%s
		  AND @q <%% sd.document
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
