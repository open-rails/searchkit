package search

import (
	"context"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"
	pgvector "github.com/pgvector/pgvector-go"
)

type Hit struct {
	EntityType string
	EntityID   string
	Model      string
	Similarity float32
}

type Options struct {
	// One or more entity types to include. Empty means "all types".
	EntityTypes []string

	// Exclude entity IDs (applied regardless of entity_type).
	ExcludeIDs []string

	// Minimum similarity threshold (cosine similarity in [0..1] typically).
	MinSimilarity float32

	// Enable two-stage retrieval (binary quantize oversample + halfvec rescore).
	TwoStage bool

	// OversampleFactor controls how many candidates stage-1 pulls vs final limit.
	// Only used when TwoStage=true. Defaults to 5.
	OversampleFactor int

	// ExtraWhereSQL is appended to WHERE (advanced escape hatch; must be SQL-safe).
	// Example: "AND entity_type <> 'gallery'".
	ExtraWhereSQL string

	// ExtraArgs are appended to the query args after the standard args.
	ExtraArgs []any
}

type Query struct {
	Schema     string
	Model      string
	QueryVec   []float32
	Limit      int
	Dimensions int // required for TwoStage; defaults to len(QueryVec) when 0
	Options    Options
}

func quoteIdent(ident string) (string, error) {
	ident = strings.TrimSpace(ident)
	if ident == "" {
		return "", fmt.Errorf("empty identifier")
	}
	for _, r := range ident {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' {
			continue
		}
		return "", fmt.Errorf("invalid identifier %q", ident)
	}
	return `"` + ident + `"`, nil
}

// SearchVectors runs a semantic KNN search against the embeddingkit-owned
// `<schema>.embedding_vectors` table and returns only candidate IDs + scores.
//
// This function intentionally does not hydrate domain rows or apply business
// logic beyond basic filtering options.
func SearchVectors(ctx context.Context, pool *pgxpool.Pool, q Query) ([]Hit, error) {
	if pool == nil {
		return nil, fmt.Errorf("pool is required")
	}
	if strings.TrimSpace(q.Schema) == "" {
		return nil, fmt.Errorf("schema is required")
	}
	if strings.TrimSpace(q.Model) == "" {
		return nil, fmt.Errorf("model is required")
	}
	if q.Limit <= 0 {
		return []Hit{}, nil
	}
	if len(q.QueryVec) == 0 {
		return []Hit{}, nil
	}

	dim := q.Dimensions
	if dim <= 0 {
		dim = len(q.QueryVec)
	}

	quotedSchema, err := quoteIdent(q.Schema)
	if err != nil {
		return nil, fmt.Errorf("invalid schema: %w", err)
	}

	half := fmt.Sprintf("halfvec(%d)", dim)
	table := quotedSchema + ".embedding_vectors"

	opts := q.Options
	if opts.OversampleFactor <= 1 {
		opts.OversampleFactor = 5
	}

	vec := pgvector.NewHalfVector(q.QueryVec)

	var sql string
	var args []any

	// Common WHERE filters.
	where := "WHERE model = $1 AND embedding IS NOT NULL"
	args = append(args, q.Model)

	argN := 2
	if len(opts.EntityTypes) > 0 {
		where += fmt.Sprintf(" AND entity_type = ANY($%d::text[])", argN)
		args = append(args, opts.EntityTypes)
		argN++
	}
	if len(opts.ExcludeIDs) > 0 {
		where += fmt.Sprintf(" AND entity_id <> ALL($%d::text[])", argN)
		args = append(args, opts.ExcludeIDs)
		argN++
	}
	if strings.TrimSpace(opts.ExtraWhereSQL) != "" {
		where += " " + strings.TrimSpace(opts.ExtraWhereSQL)
		if len(opts.ExtraArgs) > 0 {
			args = append(args, opts.ExtraArgs...)
		}
	}

	if !opts.TwoStage {
		// 1-stage cosine KNN:
		// similarity = 1 - cosine_distance
		// order by cosine_distance
		sql = fmt.Sprintf(`
			SELECT
				entity_type,
				entity_id,
				model,
				(1 - (embedding::%s <=> ($%d::%s)))::float4 AS similarity
			FROM %s
			%s
			ORDER BY embedding::%s <=> ($%d::%s)
			LIMIT $%d
		`, half, argN, half, table, where, half, argN, half, argN+1)

		args = append(args, vec, q.Limit)
	} else {
		oversample := q.Limit * opts.OversampleFactor

		// 2-stage:
		//  - stage 1: approx retrieval using binary quantize (Hamming distance)
		//  - stage 2: rescore by cosine distance
		sql = fmt.Sprintf(`
			WITH candidates AS (
				SELECT
					entity_type,
					entity_id,
					model,
					embedding
				FROM %s
				%s
				ORDER BY (binary_quantize(embedding::%s)::bit(%d)) <~> (binary_quantize($%d::%s)::bit(%d))
				LIMIT $%d
			)
			SELECT
				entity_type,
				entity_id,
				model,
				(1 - (embedding::%s <=> ($%d::%s)))::float4 AS similarity
			FROM candidates
			WHERE (1 - (embedding::%s <=> ($%d::%s))) >= $%d
			ORDER BY embedding::%s <=> ($%d::%s)
			LIMIT $%d
		`, table, where, half, dim, argN, half, dim, argN+1, half, argN+2, half, half, argN+2, half, argN+3, half, argN+2, half, argN+4)

		args = append(args,
			vec,        // $argN
			oversample, // $argN+1
			vec,        // $argN+2
			opts.MinSimilarity,
			q.Limit,
		)
	}

	rows, err := pool.Query(ctx, sql, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []Hit
	for rows.Next() {
		var h Hit
		if err := rows.Scan(&h.EntityType, &h.EntityID, &h.Model, &h.Similarity); err != nil {
			return nil, err
		}
		if opts.MinSimilarity > 0 && h.Similarity < opts.MinSimilarity {
			continue
		}
		out = append(out, h)
	}
	return out, rows.Err()
}

// SimilarTo returns nearest neighbors to an existing stored vector for the same
// model, excluding the source entity itself.
func SimilarTo(ctx context.Context, pool *pgxpool.Pool, schema string, entityType string, entityID string, model string, limit int, opts Options) ([]Hit, error) {
	if pool == nil {
		return nil, fmt.Errorf("pool is required")
	}
	if strings.TrimSpace(schema) == "" {
		return nil, fmt.Errorf("schema is required")
	}
	if strings.TrimSpace(entityType) == "" || strings.TrimSpace(entityID) == "" {
		return nil, fmt.Errorf("entityType and entityID are required")
	}
	if strings.TrimSpace(model) == "" {
		return nil, fmt.Errorf("model is required")
	}
	if limit <= 0 {
		return []Hit{}, nil
	}

	quotedSchema, err := quoteIdent(schema)
	if err != nil {
		return nil, fmt.Errorf("invalid schema: %w", err)
	}

	table := quotedSchema + ".embedding_vectors"

	where := `
		WHERE ev.model = $3
		  AND ev.embedding IS NOT NULL
		  AND NOT (ev.entity_type = $1 AND ev.entity_id = $2)
	`
	args := []any{entityType, entityID, model, limit}

	argN := 5
	if len(opts.EntityTypes) > 0 {
		where += fmt.Sprintf(" AND ev.entity_type = ANY($%d::text[])\n", argN)
		args = append(args, opts.EntityTypes)
		argN++
	}
	if len(opts.ExcludeIDs) > 0 {
		where += fmt.Sprintf(" AND ev.entity_id <> ALL($%d::text[])\n", argN)
		args = append(args, opts.ExcludeIDs)
		argN++
	}
	if strings.TrimSpace(opts.ExtraWhereSQL) != "" {
		where += " " + strings.TrimSpace(opts.ExtraWhereSQL) + "\n"
		if len(opts.ExtraArgs) > 0 {
			args = append(args, opts.ExtraArgs...)
		}
	}

	// NOTE: SimilarTo always runs 1-stage cosine KNN. Callers can run TwoStage by
	// fetching the source vector and calling SearchVectors with TwoStage=true.
	sql := fmt.Sprintf(`
		WITH source AS (
			SELECT embedding
			FROM %s
			WHERE entity_type = $1 AND entity_id = $2 AND model = $3 AND embedding IS NOT NULL
			LIMIT 1
		)
		SELECT
			ev.entity_type,
			ev.entity_id,
			ev.model,
			(1 - (ev.embedding <=> s.embedding))::float4 AS similarity
		FROM %s ev, source s
		%s
		ORDER BY ev.embedding <=> s.embedding
		LIMIT $4
	`, table, table, where)

	rows, err := pool.Query(ctx, sql, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []Hit
	for rows.Next() {
		var h Hit
		if err := rows.Scan(&h.EntityType, &h.EntityID, &h.Model, &h.Similarity); err != nil {
			return nil, err
		}
		if opts.MinSimilarity > 0 && h.Similarity < opts.MinSimilarity {
			continue
		}
		out = append(out, h)
	}
	return out, rows.Err()
}
