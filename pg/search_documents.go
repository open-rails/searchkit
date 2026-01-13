package pg

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/doujins-org/searchkit/internal/textnormalize"
	"github.com/jackc/pgx/v5/pgxpool"
)

const searchDocumentsTable = "search_documents"

// UpsertSearchDocuments upserts lexical (trigram) documents for one (entity_type, language).
//
// Documents are heavy-normalized by searchkit before storage so host apps can pass
// "raw-ish" display strings.
func UpsertSearchDocuments(ctx context.Context, pool *pgxpool.Pool, schema string, entityType string, language string, docs map[string]string) error {
	if pool == nil {
		return fmt.Errorf("pool is required")
	}
	if strings.TrimSpace(schema) == "" {
		return fmt.Errorf("schema is required")
	}
	if strings.TrimSpace(entityType) == "" {
		return fmt.Errorf("entityType is required")
	}
	if strings.TrimSpace(language) == "" {
		return fmt.Errorf("language is required")
	}
	if len(docs) == 0 {
		return nil
	}

	qs, err := quoteIdent(schema)
	if err != nil {
		return fmt.Errorf("invalid schema: %w", err)
	}

	// Stable ordering helps tests/debugging and keeps array ordering deterministic.
	ids := make([]string, 0, len(docs))
	for id := range docs {
		if strings.TrimSpace(id) == "" {
			continue
		}
		ids = append(ids, id)
	}
	sort.Strings(ids)
	if len(ids) == 0 {
		return nil
	}

	idArr := make([]string, 0, len(ids))
	docArr := make([]string, 0, len(ids))
	rawArr := make([]string, 0, len(ids))
	var deleteIDs []string
	for _, id := range ids {
		raw := docs[id]
		rawTrim := strings.TrimSpace(raw)
		norm := strings.TrimSpace(textnormalize.Heavy(rawTrim))
		if norm == "" {
			deleteIDs = append(deleteIDs, id)
			continue
		}
		idArr = append(idArr, id)
		docArr = append(docArr, norm)
		if rawTrim == "" {
			rawTrim = norm
		}
		rawArr = append(rawArr, rawTrim)
	}

	if len(idArr) > 0 {
		q := fmt.Sprintf(`
			WITH rows AS (
				SELECT
					unnest($3::text[]) AS entity_id,
					unnest($4::text[]) AS raw_document,
					unnest($5::text[]) AS document
			)
			INSERT INTO %s.%s (entity_type, entity_id, language, raw_document, document, tsv, created_at, updated_at)
			SELECT
				$1,
				rows.entity_id,
				$2,
				rows.raw_document,
				rows.document,
				to_tsvector(%s.searchkit_regconfig_for_language($2), rows.raw_document),
				now(),
				now()
			FROM rows
			ON CONFLICT (entity_type, entity_id, language) DO UPDATE SET
				raw_document = EXCLUDED.raw_document,
				document = EXCLUDED.document,
				tsv = EXCLUDED.tsv,
				updated_at = now()
		`, qs, searchDocumentsTable, qs)
		if _, err := pool.Exec(ctx, q, entityType, language, idArr, rawArr, docArr); err != nil {
			return err
		}
	}

	if len(deleteIDs) > 0 {
		q := fmt.Sprintf(`
			DELETE FROM %s.%s
			WHERE entity_type = $1 AND language = $2 AND entity_id = ANY($3::text[])
		`, qs, searchDocumentsTable)
		if _, err := pool.Exec(ctx, q, entityType, language, deleteIDs); err != nil {
			return err
		}
	}

	return nil
}

func DeleteSearchDocuments(ctx context.Context, pool *pgxpool.Pool, schema string, entityType string, entityID string, language string) error {
	return DeleteSearchDocumentsMany(ctx, pool, schema, entityType, []string{entityID}, language)
}

func DeleteSearchDocumentsMany(ctx context.Context, pool *pgxpool.Pool, schema string, entityType string, entityIDs []string, language string) error {
	if pool == nil {
		return fmt.Errorf("pool is required")
	}
	if strings.TrimSpace(schema) == "" {
		return fmt.Errorf("schema is required")
	}
	if strings.TrimSpace(entityType) == "" {
		return fmt.Errorf("entityType is required")
	}
	if strings.TrimSpace(language) == "" {
		return fmt.Errorf("language is required")
	}
	if len(entityIDs) == 0 {
		return nil
	}
	qs, err := quoteIdent(schema)
	if err != nil {
		return fmt.Errorf("invalid schema: %w", err)
	}
	q := fmt.Sprintf(`
		DELETE FROM %s.%s
		WHERE entity_type = $1 AND language = $2 AND entity_id = ANY($3::text[])
	`, qs, searchDocumentsTable)
	_, err = pool.Exec(ctx, q, entityType, language, entityIDs)
	return err
}
