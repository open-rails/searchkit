package pg

import (
	"context"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"
)

// DeleteEmbeddingVectorsForEntity deletes all embeddings (all models) for an entity+language.
func DeleteEmbeddingVectorsForEntity(ctx context.Context, pool *pgxpool.Pool, schema string, entityType string, entityID string, language string) error {
	if pool == nil {
		return fmt.Errorf("pool is required")
	}
	if strings.TrimSpace(schema) == "" {
		return fmt.Errorf("schema is required")
	}
	if strings.TrimSpace(entityType) == "" || strings.TrimSpace(entityID) == "" || strings.TrimSpace(language) == "" {
		return nil
	}
	qs, err := quoteIdent(schema)
	if err != nil {
		return fmt.Errorf("invalid schema: %w", err)
	}
	q := fmt.Sprintf(`
		DELETE FROM %s.embedding_vectors
		WHERE entity_type = $1 AND entity_id = $2 AND language = $3
	`, qs)
	_, err = pool.Exec(ctx, q, entityType, entityID, language)
	return err
}

// FilterMissingEmbeddings returns the subset of entityIDs that do NOT currently
// have an embedding vector for (entity_type, model, language).
func FilterMissingEmbeddings(ctx context.Context, pool *pgxpool.Pool, schema string, entityType string, model string, language string, entityIDs []string) ([]string, error) {
	if pool == nil {
		return nil, fmt.Errorf("pool is required")
	}
	if strings.TrimSpace(schema) == "" {
		return nil, fmt.Errorf("schema is required")
	}
	if strings.TrimSpace(entityType) == "" || strings.TrimSpace(model) == "" || strings.TrimSpace(language) == "" {
		return nil, fmt.Errorf("entityType, model, and language are required")
	}
	if len(entityIDs) == 0 {
		return nil, nil
	}
	qs, err := quoteIdent(schema)
	if err != nil {
		return nil, fmt.Errorf("invalid schema: %w", err)
	}
	q := fmt.Sprintf(`
		WITH ids AS (
			SELECT unnest($4::text[]) AS entity_id
		)
		SELECT ids.entity_id
		FROM ids
		LEFT JOIN %s.embedding_vectors ev
			ON ev.entity_type = $1
			AND ev.entity_id = ids.entity_id
			AND ev.model = $2
			AND ev.language = $3
		WHERE ev.entity_id IS NULL
	`, qs)
	rows, err := pool.Query(ctx, q, entityType, model, language, entityIDs)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		if strings.TrimSpace(id) != "" {
			out = append(out, id)
		}
	}
	return out, rows.Err()
}
