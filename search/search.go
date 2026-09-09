package search

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	pgvector "github.com/pgvector/pgvector-go"
)

type Hit struct {
	EntityType string
	EntityID   string
	Model      string
	Language   string
	Similarity float32
}

type Options struct {
	// One or more entity types to include. Empty means "all types".
	EntityTypes []string

	// Exclude entity IDs (applied regardless of entity_type).
	ExcludeIDs []string

	// Minimum similarity threshold (cosine similarity in [0..1] typically).
	MinSimilarity float32
	// MinSimilarityEnabled applies MinSimilarity even when it is zero.
	MinSimilarityEnabled bool

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

const (
	MaxCandidateLimit = 10_000
)

type Query struct {
	Schema     string
	Model      string
	Language   string
	QueryVec   []float32
	Limit      int
	Dimensions int // required for TwoStage; defaults to len(QueryVec) when 0
	Options    Options
}

// EffectiveOversampleFactor returns the factor used by two-stage retrieval.
func EffectiveOversampleFactor(factor int) int {
	if factor <= 1 {
		return 5
	}
	return factor
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

// SemanticSearch runs a semantic KNN search against the searchkit-owned
// `<schema>.embedding_vectors` table and returns only candidate IDs + scores.
//
// This function intentionally does not hydrate domain rows or apply business
// logic beyond basic filtering options.
func SemanticSearch(ctx context.Context, pool *pgxpool.Pool, q Query) ([]Hit, error) {
	if math.IsNaN(float64(q.Options.MinSimilarity)) || math.IsInf(float64(q.Options.MinSimilarity), 0) {
		return nil, fmt.Errorf("min similarity must be finite")
	}
	if pool == nil {
		return nil, fmt.Errorf("pool is required")
	}
	if strings.TrimSpace(q.Schema) == "" {
		return nil, fmt.Errorf("schema is required")
	}
	if strings.TrimSpace(q.Model) == "" {
		return nil, fmt.Errorf("model is required")
	}
	if strings.TrimSpace(q.Language) == "" {
		return nil, fmt.Errorf("language is required")
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
	if opts.MinSimilarity <= 0 && !opts.MinSimilarityEnabled {
		opts.MinSimilarity = 0
	}
	opts.OversampleFactor = EffectiveOversampleFactor(opts.OversampleFactor)
	applyMinSimilarity := opts.MinSimilarityEnabled || opts.MinSimilarity > 0
	oversample := 0
	if opts.TwoStage {
		if q.Limit > int(^uint(0)>>1)/opts.OversampleFactor {
			return nil, fmt.Errorf("candidate oversample limit overflows int")
		}
		oversample = q.Limit * opts.OversampleFactor
	}

	vec := pgvector.NewHalfVector(q.QueryVec)

	var sql string
	args := pgx.NamedArgs{
		"model":    q.Model,
		"language": q.Language,
		"qvec":     vec,
		"limit":    q.Limit,
	}
	if opts.TwoStage {
		args["oversample"] = oversample
	}

	// Common WHERE filters.
	where := "WHERE ev.model = @model AND ev.language = @language AND ev.embedding IS NOT NULL"
	if applyMinSimilarity {
		args["min_similarity"] = opts.MinSimilarity
	}
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
	if !opts.TwoStage && applyMinSimilarity {
		where += fmt.Sprintf(" AND (1 - (ev.embedding::%s <=> (@qvec::%s))) >= @min_similarity", half, half)
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
				ev.language,
				(1 - (ev.embedding::%s <=> (@qvec::%s)))::float4 AS similarity
			FROM %s ev
			%s
			ORDER BY ev.embedding::%s <=> (@qvec::%s), ev.entity_type, ev.entity_id, ev.language, ev.model
			LIMIT @limit
		`, half, half, table, where, half, half)

	} else {
		minSimilarityFilter := ""
		if applyMinSimilarity {
			minSimilarityFilter = "WHERE (1 - (embedding::%s <=> (@qvec::%s))) >= @min_similarity"
			minSimilarityFilter = fmt.Sprintf(minSimilarityFilter, half, half)
		}

		// 2-stage:
		//  - stage 1: approx retrieval using binary quantize (Hamming distance)
		//  - stage 2: rescore by cosine distance
		sql = fmt.Sprintf(`
				WITH candidates AS (
					SELECT
						ev.entity_type,
						ev.entity_id,
						ev.model,
						ev.language,
						ev.embedding
					FROM %s ev
					%s
					ORDER BY (binary_quantize(embedding::%s)::bit(%d)) <~> (binary_quantize(@qvec::%s)::bit(%d)), ev.entity_type, ev.entity_id, ev.language, ev.model
					LIMIT @oversample
				)
				SELECT
					entity_type,
					entity_id,
					model,
					language,
					(1 - (embedding::%s <=> (@qvec::%s)))::float4 AS similarity
				FROM candidates
				%s
				ORDER BY embedding::%s <=> (@qvec::%s), entity_type, entity_id, language, model
				LIMIT @limit
			`, table, where, half, dim, half, dim, half, half, minSimilarityFilter, half, half)

	}

	rows, err := pool.Query(ctx, sql, args)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []Hit
	for rows.Next() {
		var h Hit
		if err := rows.Scan(&h.EntityType, &h.EntityID, &h.Model, &h.Language, &h.Similarity); err != nil {
			return nil, err
		}
		out = append(out, h)
	}
	return out, rows.Err()
}

// SimilarTo returns nearest neighbors to an existing stored vector for the same
// model, excluding the source entity itself.
func SimilarTo(ctx context.Context, pool *pgxpool.Pool, schema string, entityType string, entityID string, model string, language string, limit int, opts Options) ([]Hit, error) {
	if math.IsNaN(float64(opts.MinSimilarity)) || math.IsInf(float64(opts.MinSimilarity), 0) {
		return nil, fmt.Errorf("min similarity must be finite")
	}
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
	if strings.TrimSpace(language) == "" {
		return nil, fmt.Errorf("language is required")
	}
	if limit <= 0 {
		return []Hit{}, nil
	}

	quotedSchema, err := quoteIdent(schema)
	if err != nil {
		return nil, fmt.Errorf("invalid schema: %w", err)
	}

	table := quotedSchema + ".embedding_vectors"

	// The source vector is read first and then bound as a parameter. Joining it
	// in as a CTE column made the ORDER BY a join condition, which no HNSW index
	// can answer, so every "more like this" walked the whole corpus computing
	// distances — at a cost set by the corpus rather than by the limit asked for.
	var (
		sourceVec  []float32
		sourceDims int
	)
	err = pool.QueryRow(ctx, `
		SELECT embedding::real[], vector_dims(embedding)
		FROM `+table+`
		WHERE entity_type = $1 AND entity_id = $2 AND model = $3 AND language = $4
		  AND embedding IS NOT NULL
		LIMIT 1
	`, entityType, entityID, model, language).Scan(&sourceVec, &sourceDims)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			// No embedding for the source: nothing is similar to it, which is
			// what the joined form returned too.
			return []Hit{}, nil
		}
		return nil, err
	}
	if sourceDims <= 0 || len(sourceVec) == 0 {
		return []Hit{}, nil
	}
	half := fmt.Sprintf("halfvec(%d)", sourceDims)

	where := `
		WHERE ev.model = @model
		  AND ev.language = @language
		  AND ev.embedding IS NOT NULL
		  AND NOT (ev.entity_type = @entity_type AND ev.entity_id = @entity_id)
	`
	args := pgx.NamedArgs{
		"entity_type": entityType,
		"entity_id":   entityID,
		"model":       model,
		"language":    language,
		"limit":       limit,
		"qvec":        pgvector.NewHalfVector(sourceVec),
	}
	if opts.MinSimilarityEnabled || opts.MinSimilarity > 0 {
		args["min_similarity"] = opts.MinSimilarity
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
	if opts.MinSimilarityEnabled || opts.MinSimilarity > 0 {
		where += fmt.Sprintf(" AND (1 - (ev.embedding::%s <=> (@qvec::%s))) >= @min_similarity\n", half, half)
	}

	// NOTE: SimilarTo always runs 1-stage cosine KNN. Callers can run TwoStage by
	// fetching the source vector and calling SearchVectors with TwoStage=true.
	// The ORDER BY casts to halfvec(dims) so it matches the per-model HNSW
	// expression index; the trailing keys only break ties, which Postgres
	// resolves with an incremental sort on top of the index scan.
	sql := fmt.Sprintf(`
		SELECT
			ev.entity_type,
			ev.entity_id,
			ev.model,
			ev.language,
			(1 - (ev.embedding::%s <=> (@qvec::%s)))::float4 AS similarity
		FROM %s ev
		%s
		ORDER BY ev.embedding::%s <=> (@qvec::%s), ev.entity_type, ev.entity_id, ev.language, ev.model
		LIMIT @limit
	`, half, half, table, where, half, half)

	// An HNSW scan only visits ef_search candidates, 40 by default — a caller
	// asking for more than that would quietly get fewer neighbours than the old
	// full scan returned. Widen the candidate list for this statement alone,
	// inside a transaction, so SET LOCAL reverts with it and no pooled
	// connection keeps the setting.
	tx, err := pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	if _, err := tx.Exec(ctx, fmt.Sprintf("SET LOCAL hnsw.ef_search = %d", efSearchFor(limit))); err != nil {
		return nil, err
	}

	rows, err := tx.Query(ctx, sql, args)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []Hit
	for rows.Next() {
		var h Hit
		if err := rows.Scan(&h.EntityType, &h.EntityID, &h.Model, &h.Language, &h.Similarity); err != nil {
			return nil, err
		}
		out = append(out, h)
	}
	return out, rows.Err()
}

// efSearchFor sizes the HNSW candidate list for a query that wants `limit`
// neighbours. Recall needs the list to be at least as long as the limit, and a
// little beyond it, but the scan cost grows with it — so it is bounded.
func efSearchFor(limit int) int {
	const (
		minEF = 40
		maxEF = 1000
	)
	ef := limit * 2
	if ef < minEF {
		ef = minEF
	}
	if ef > maxEF {
		ef = maxEF
	}
	return ef
}
