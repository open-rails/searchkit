package search

import (
	"context"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5"
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

	// FilterSQL is an optional additional WHERE fragment appended to the query as:
	//   ... AND (<FilterSQL>)
	//
	// It is intended for app-owned constraints (e.g. language availability) that
	// must be enforced inside the KNN query.
	//
	// IMPORTANT: this is trusted SQL provided by the host app. Do not insert
	// user input into it unsafely.
	FilterSQL string
	// FilterArgs are named args referenced by FilterSQL using pgx '@name'
	// placeholders (e.g. "... language = @lang").
	FilterArgs map[string]any
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

func mergeNamedArgs(dst pgx.NamedArgs, extra map[string]any) error {
	if len(extra) == 0 {
		return nil
	}
	for k, v := range extra {
		k = strings.TrimSpace(k)
		if k == "" {
			return fmt.Errorf("empty FilterArgs key")
		}
		if _, exists := dst[k]; exists {
			return fmt.Errorf("FilterArgs key %q conflicts with reserved arg", k)
		}
		dst[k] = v
	}
	return nil
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
	args := pgx.NamedArgs{}

	// Common WHERE filters.
	where := "WHERE ev.model = @model AND ev.embedding IS NOT NULL"
	args["model"] = q.Model
	if len(opts.EntityTypes) > 0 {
		where += " AND ev.entity_type = ANY(@entity_types::text[])"
		args["entity_types"] = opts.EntityTypes
	}
	if len(opts.ExcludeIDs) > 0 {
		where += " AND ev.entity_id <> ALL(@exclude_ids::text[])"
		args["exclude_ids"] = opts.ExcludeIDs
	}
	if strings.TrimSpace(opts.FilterSQL) != "" {
		where += " AND (" + opts.FilterSQL + ")"
		if err := mergeNamedArgs(args, opts.FilterArgs); err != nil {
			return nil, err
		}
	}

	if !opts.TwoStage {
		// 1-stage cosine KNN:
		// similarity = 1 - cosine_distance
		// order by cosine_distance
		sql = fmt.Sprintf(`
			SELECT
				ev.entity_type,
				ev.entity_id,
				ev.model,
				(1 - (ev.embedding::%s <=> (@qvec::%s)))::float4 AS similarity
			FROM %s ev
			%s
			ORDER BY ev.embedding::%s <=> (@qvec::%s)
			LIMIT @limit
		`, half, half, table, where, half, half)

		args["qvec"] = vec
		args["limit"] = q.Limit
	} else {
		oversample := q.Limit * opts.OversampleFactor

		// 2-stage:
		//  - stage 1: approx retrieval using binary quantize (Hamming distance)
		//  - stage 2: rescore by cosine distance
		sql = fmt.Sprintf(`
				WITH candidates AS (
					SELECT
						ev.entity_type,
						ev.entity_id,
						ev.model,
						ev.embedding
					FROM %s ev
					%s
					ORDER BY (binary_quantize(embedding::%s)::bit(%d)) <~> (binary_quantize(@qvec::%s)::bit(%d))
					LIMIT @oversample
				)
				SELECT
					entity_type,
					entity_id,
					model,
					(1 - (embedding::%s <=> (@qvec::%s)))::float4 AS similarity
				FROM candidates
				WHERE (1 - (embedding::%s <=> (@qvec::%s))) >= @min_similarity
				ORDER BY embedding::%s <=> (@qvec::%s)
				LIMIT @limit
			`, table, where, half, dim, half, dim, half, half, half, half, half, half)

		args["qvec"] = vec
		args["oversample"] = oversample
		args["min_similarity"] = opts.MinSimilarity
		args["limit"] = q.Limit
	}

	rows, err := pool.Query(ctx, sql, args)
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
		WHERE ev.model = @model
		  AND ev.embedding IS NOT NULL
		  AND NOT (ev.entity_type = @entity_type AND ev.entity_id = @entity_id)
	`
	args := pgx.NamedArgs{
		"entity_type": entityType,
		"entity_id":   entityID,
		"model":       model,
		"limit":       limit,
	}

	if len(opts.EntityTypes) > 0 {
		where += " AND ev.entity_type = ANY(@entity_types::text[])\n"
		args["entity_types"] = opts.EntityTypes
	}
	if len(opts.ExcludeIDs) > 0 {
		where += " AND ev.entity_id <> ALL(@exclude_ids::text[])\n"
		args["exclude_ids"] = opts.ExcludeIDs
	}
	if strings.TrimSpace(opts.FilterSQL) != "" {
		where += " AND (" + opts.FilterSQL + ")\n"
		if err := mergeNamedArgs(args, opts.FilterArgs); err != nil {
			return nil, err
		}
	}

	// NOTE: SimilarTo always runs 1-stage cosine KNN. Callers can run TwoStage by
	// fetching the source vector and calling SearchVectors with TwoStage=true.
	sql := fmt.Sprintf(`
		WITH source AS (
			SELECT embedding
			FROM %s
			WHERE entity_type = @entity_type AND entity_id = @entity_id AND model = @model AND embedding IS NOT NULL
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
		LIMIT @limit
	`, table, table, where)

	rows, err := pool.Query(ctx, sql, args)
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
