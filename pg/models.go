package pg

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"
)

type ModelSpec struct {
	Name     string // stored in embedding_models.model
	Dims     int    // fixed dims for the model
	Modality string // "text" | "vl"
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

func quoteLiteral(s string) string {
	return "'" + strings.ReplaceAll(s, "'", "''") + "'"
}

func indexSuffix(model string, dims int) string {
	h := sha1.Sum([]byte(fmt.Sprintf("%s:%d", model, dims)))
	return hex.EncodeToString(h[:8])
}

// UpsertModels syncs the configured model specs into `<schema>.embedding_models`.
func UpsertModels(ctx context.Context, pool *pgxpool.Pool, schema string, models []ModelSpec) error {
	if pool == nil {
		return fmt.Errorf("pool is required")
	}
	qs, err := quoteIdent(schema)
	if err != nil {
		return fmt.Errorf("invalid schema: %w", err)
	}

	// Treat `models` as the active configured set. We upsert everything provided,
	// then prune any rows (and related state) for models that are no longer active.
	var active []string
	for _, m := range models {
		name := strings.TrimSpace(m.Name)
		if name == "" {
			return fmt.Errorf("model name is required")
		}
		if m.Dims <= 0 {
			return fmt.Errorf("model %q dims must be > 0", name)
		}
		modality := strings.TrimSpace(m.Modality)
		if modality == "" {
			return fmt.Errorf("model %q modality is required", name)
		}

		q := fmt.Sprintf(`
			INSERT INTO %s.embedding_models (model, dims, modality, created_at, updated_at)
			VALUES ($1, $2, $3, now(), now())
			ON CONFLICT (model) DO UPDATE SET
				dims = EXCLUDED.dims,
				modality = EXCLUDED.modality,
				updated_at = now()
		`, qs)
		if _, err := pool.Exec(ctx, q, name, m.Dims, modality); err != nil {
			return err
		}

		active = append(active, name)
	}

	// Prune removed models (and their auxiliary state) so `embedding_models`
	// represents only active configured models.
	//
	// NOTE: We intentionally do NOT delete from embedding_vectors here; that data
	// can be large and is not required for correctness (search won’t use removed
	// models if the host config no longer references them).
	qPruneModels := fmt.Sprintf(`
		DELETE FROM %s.embedding_models
		WHERE NOT (model = ANY($1::text[]))
	`, qs)
	if _, err := pool.Exec(ctx, qPruneModels, active); err != nil {
		return err
	}

	qPruneTasks := fmt.Sprintf(`
		DELETE FROM %s.embedding_tasks
		WHERE NOT (model = ANY($1::text[]))
	`, qs)
	if _, err := pool.Exec(ctx, qPruneTasks, active); err != nil {
		return err
	}

	qPruneBackfill := fmt.Sprintf(`
		DELETE FROM %s.embedding_vectors_backfill_state
		WHERE NOT (model = ANY($1::text[]))
	`, qs)
	if _, err := pool.Exec(ctx, qPruneBackfill, active); err != nil {
		return err
	}

	qPruneDLQ := fmt.Sprintf(`
		DELETE FROM %s.embedding_dead_letters
		WHERE NOT (model = ANY($1::text[]))
	`, qs)
	if _, err := pool.Exec(ctx, qPruneDLQ, active); err != nil {
		return err
	}

	return nil
}

// EnsureModelIndexes creates per-model partial HNSW indexes for:
//   - cosine distance (1-stage)
//   - binary quantize + Hamming distance (2-stage stage-1)
//
// This must NOT run inside a transaction because it uses CREATE INDEX CONCURRENTLY.
func EnsureModelIndexes(ctx context.Context, pool *pgxpool.Pool, schema string, model string, dims int) error {
	if pool == nil {
		return fmt.Errorf("pool is required")
	}
	qs, err := quoteIdent(schema)
	if err != nil {
		return fmt.Errorf("invalid schema: %w", err)
	}
	model = strings.TrimSpace(model)
	if model == "" {
		return fmt.Errorf("model is required")
	}
	if dims <= 0 {
		return fmt.Errorf("dims must be > 0")
	}

	// NOTE: We intentionally cast embedding to halfvec(dims) inside the index
	// expression so each model index has fixed dimensions.
	half := fmt.Sprintf("halfvec(%d)", dims)
	pred := "model = " + quoteLiteral(model) + " AND embedding IS NOT NULL"

	suffix := indexSuffix(model, dims)
	cosIdx := fmt.Sprintf("idx_embedding_vectors_hnsw_cosine__%s", suffix)
	binIdx := fmt.Sprintf("idx_embedding_vectors_hnsw_binary__%s", suffix)

	// 1) Cosine HNSW (expression index).
	q1 := fmt.Sprintf(`
		CREATE INDEX CONCURRENTLY IF NOT EXISTS %s
		ON %s.embedding_vectors
		USING hnsw ((embedding::%s) halfvec_cosine_ops)
		WHERE %s
	`, cosIdx, qs, half, pred)
	if _, err := pool.Exec(ctx, q1); err != nil {
		return err
	}

	// 2) Binary HNSW for two-stage retrieval (expression index).
	// binary_quantize(halfvec) -> bit(dims); <~> is Hamming distance.
	q2 := fmt.Sprintf(`
		CREATE INDEX CONCURRENTLY IF NOT EXISTS %s
		ON %s.embedding_vectors
		USING hnsw ((binary_quantize(embedding::%s)::bit(%d)) bit_hamming_ops)
		WHERE %s
	`, binIdx, qs, half, dims, pred)
	if _, err := pool.Exec(ctx, q2); err != nil {
		return err
	}

	return nil
}

// EnsureIndexesForModels ensures per-model cosine+binary indexes for every model spec.
func EnsureIndexesForModels(ctx context.Context, pool *pgxpool.Pool, schema string, models []ModelSpec) error {
	for _, m := range models {
		if err := EnsureModelIndexes(ctx, pool, schema, m.Name, m.Dims); err != nil {
			return err
		}
	}
	return nil
}
