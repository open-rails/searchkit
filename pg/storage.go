package pg

import (
	"context"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"
	pgvector "github.com/pgvector/pgvector-go"

	"github.com/doujins-org/embeddingkit/runtime"
	"github.com/doujins-org/embeddingkit/vl"
)

const embeddingVectorsTable = "embedding_vectors"

// PostgresStorage is a reference implementation of runtime.Storage that writes
// embeddings into embeddingkit-owned tables in the host application's schema.
//
// Tables:
//   - <schema>.embedding_vectors
type PostgresStorage struct {
	pool   *pgxpool.Pool
	schema string
}

var _ runtime.Storage = (*PostgresStorage)(nil)

func NewPostgresStorage(pool *pgxpool.Pool, schema string) *PostgresStorage {
	return &PostgresStorage{pool: pool, schema: schema}
}

func (s *PostgresStorage) UpsertTextEmbedding(ctx context.Context, entityType string, entityID string, model string, dim int, embedding []float32) error {
	if s.schema == "" {
		return fmt.Errorf("schema is required")
	}
	if entityType == "" || model == "" {
		return fmt.Errorf("entityType and model are required")
	}
	if strings.TrimSpace(entityID) == "" {
		return fmt.Errorf("entityID is required")
	}
	if len(embedding) == 0 {
		return fmt.Errorf("embedding is empty")
	}

	q := fmt.Sprintf(`
		INSERT INTO %s.%s (entity_type, entity_id, model, embedding, created_at, updated_at)
		VALUES ($1, $2, $3, $4, now(), now())
		ON CONFLICT (entity_type, entity_id, model) DO UPDATE SET
			embedding = EXCLUDED.embedding,
			updated_at = now()
	`, s.schema, embeddingVectorsTable)

	_, err := s.pool.Exec(ctx, q, entityType, entityID, model, pgvector.NewHalfVector(embedding))
	return err
}

func (s *PostgresStorage) UpsertVLEmbeddingAsset(ctx context.Context, entityType string, entityID string, model string, dim int, ref vl.AssetRef, embedding []float32) error {
	return fmt.Errorf("vl embedding storage not implemented")
}

func (s *PostgresStorage) UpsertVLAggregateEmbedding(ctx context.Context, entityType string, entityID string, model string, dim int, embedding []float32) error {
	// In v1, store VL embeddings in the same canonical table as text embeddings.
	return s.UpsertTextEmbedding(ctx, entityType, entityID, model, dim, embedding)
}
