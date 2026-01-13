package worker

import (
	"context"
	"fmt"
	"strings"

	"github.com/doujins-org/searchkit/pg"
	"github.com/doujins-org/searchkit/runtime"
	"github.com/doujins-org/searchkit/tasks"
	"github.com/jackc/pgx/v5/pgxpool"
)

type ListEntityIDsPage func(ctx context.Context, entityType string, language string, cursor string, limit int) (ids []string, nextCursor string, done bool, err error)

type SearchkitOptions struct {
	// Required.
	Pool   *pgxpool.Pool
	Schema string

	// Required.
	SupportedLanguages []string

	// Which entity types are lexically indexed (stored in search_documents).
	LexicalEntityTypes []string

	// Which entity types are semantically embedded (stored in embedding_vectors).
	SemanticEntityTypes []string

	// Required for backfill.
	ListEntityIDsPage ListEntityIDsPage

	// Optional overrides.
	TaskRepo *tasks.Repo

	// Batch sizing (defaults are conservative).
	DirtyBatchSize   int
	BackfillPageSize int
	// Upper bound on how much cursor backfill work to do per RunOnce.
	BackfillMaxPages int

	// Embedding task draining settings (existing embedding worker).
	DrainOptions Options
}

func (o SearchkitOptions) withDefaults() SearchkitOptions {
	out := o
	if out.DirtyBatchSize <= 0 {
		out.DirtyBatchSize = 250
	}
	if out.BackfillPageSize <= 0 {
		out.BackfillPageSize = 1000
	}
	if out.BackfillMaxPages <= 0 {
		out.BackfillMaxPages = 5
	}
	out.DrainOptions = out.DrainOptions.withDefaults()
	return out
}

type dirtyRow struct {
	EntityType string
	EntityID   string
	Language   string
	IsDeleted  bool
	Reason     string
}

func RunOnceSearchkit(ctx context.Context, rt *runtime.Runtime, opts SearchkitOptions) error {
	if rt == nil {
		return fmt.Errorf("runtime is required")
	}
	cfg := opts.withDefaults()
	if cfg.Pool == nil {
		return fmt.Errorf("pool is required")
	}
	if strings.TrimSpace(cfg.Schema) == "" {
		return fmt.Errorf("schema is required")
	}
	if len(cfg.SupportedLanguages) == 0 {
		return fmt.Errorf("SupportedLanguages is required")
	}
	if cfg.ListEntityIDsPage == nil {
		return fmt.Errorf("ListEntityIDsPage is required")
	}
	repo := cfg.TaskRepo
	if repo == nil {
		repo = tasks.NewRepo(cfg.Pool, cfg.Schema)
	}

	lexicalSet := make(map[string]struct{}, len(cfg.LexicalEntityTypes))
	for _, t := range cfg.LexicalEntityTypes {
		t = strings.TrimSpace(t)
		if t == "" {
			continue
		}
		lexicalSet[t] = struct{}{}
	}
	semanticSet := make(map[string]struct{}, len(cfg.SemanticEntityTypes))
	for _, t := range cfg.SemanticEntityTypes {
		t = strings.TrimSpace(t)
		if t == "" {
			continue
		}
		semanticSet[t] = struct{}{}
	}

	// 1) Drain dirty queue (fast path).
	if err := processDirtyOnce(ctx, cfg.Pool, cfg.Schema, repo, rt, lexicalSet, semanticSet, cfg.DirtyBatchSize); err != nil {
		return err
	}

	// 2) Bounded backfill tick (slow path).
	if err := backfillOnce(ctx, cfg.Pool, cfg.Schema, repo, rt, lexicalSet, semanticSet, cfg.SupportedLanguages, cfg.ListEntityIDsPage, cfg.BackfillPageSize, cfg.BackfillMaxPages); err != nil {
		return err
	}

	// 3) Drain embedding tasks (does the provider calls + writes embedding_vectors).
	return DrainOnce(ctx, rt, repo, cfg.DrainOptions)
}

func processDirtyOnce(
	ctx context.Context,
	pool *pgxpool.Pool,
	schema string,
	repo *tasks.Repo,
	rt *runtime.Runtime,
	lexicalSet map[string]struct{},
	semanticSet map[string]struct{},
	limit int,
) error {
	if limit <= 0 {
		return nil
	}
	qs, err := pg.QuoteSchema(schema)
	if err != nil {
		return err
	}

	rows, err := pool.Query(ctx, fmt.Sprintf(`
		SELECT entity_type, entity_id, language, is_deleted, reason
		FROM %s.search_dirty
		ORDER BY updated_at ASC
		LIMIT $1
	`, qs), limit)
	if err != nil {
		return err
	}
	defer rows.Close()

	var batch []dirtyRow
	for rows.Next() {
		var r dirtyRow
		if err := rows.Scan(&r.EntityType, &r.EntityID, &r.Language, &r.IsDeleted, &r.Reason); err != nil {
			return err
		}
		if strings.TrimSpace(r.EntityType) == "" || strings.TrimSpace(r.EntityID) == "" || strings.TrimSpace(r.Language) == "" {
			continue
		}
		batch = append(batch, r)
	}
	if err := rows.Err(); err != nil {
		return err
	}
	if len(batch) == 0 {
		return nil
	}

	// Process deletions first.
	for _, r := range batch {
		if !r.IsDeleted {
			continue
		}
		if err := pg.DeleteSearchDocuments(ctx, pool, schema, r.EntityType, r.EntityID, r.Language); err != nil {
			return err
		}
		if err := pg.DeleteEmbeddingVectorsForEntity(ctx, pool, schema, r.EntityType, r.EntityID, r.Language); err != nil {
			return err
		}
		if err := repo.DeleteAllForEntity(ctx, r.EntityType, r.EntityID, r.Language); err != nil {
			return err
		}
	}

	// Lexical updates.
	groupedLex := make(map[string]map[string][]string) // entity_type -> language -> ids
	for _, r := range batch {
		if r.IsDeleted {
			continue
		}
		if _, ok := lexicalSet[r.EntityType]; !ok {
			continue
		}
		if groupedLex[r.EntityType] == nil {
			groupedLex[r.EntityType] = make(map[string][]string)
		}
		groupedLex[r.EntityType][r.Language] = append(groupedLex[r.EntityType][r.Language], r.EntityID)
	}
	for et, byLang := range groupedLex {
		for lang, ids := range byLang {
			docs, err := rt.BuildLexicalString(ctx, et, lang, ids)
			if err != nil {
				return err
			}
			if err := pg.UpsertSearchDocuments(ctx, pool, schema, et, lang, docs); err != nil {
				return err
			}
		}
	}

	// Semantic: enqueue tasks for all active models (no need to build docs here).
	activeModels := rt.ActiveModels()
	groupedSem := make(map[string]map[string][]string) // entity_type -> language -> ids
	for _, r := range batch {
		if r.IsDeleted {
			continue
		}
		if _, ok := semanticSet[r.EntityType]; !ok {
			continue
		}
		if groupedSem[r.EntityType] == nil {
			groupedSem[r.EntityType] = make(map[string][]string)
		}
		groupedSem[r.EntityType][r.Language] = append(groupedSem[r.EntityType][r.Language], r.EntityID)
	}
	for et, byLang := range groupedSem {
		for lang, ids := range byLang {
			for _, model := range activeModels {
				if err := repo.EnqueueMany(ctx, et, ids, model, lang, "dirty"); err != nil {
					return err
				}
			}
		}
	}

	// Clear dirty rows (processed).
	tx, err := pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	for _, r := range batch {
		if _, err := tx.Exec(ctx, fmt.Sprintf(`
			DELETE FROM %s.search_dirty
			WHERE entity_type = $1 AND entity_id = $2 AND language = $3
		`, qs), r.EntityType, r.EntityID, r.Language); err != nil {
			return err
		}
	}
	return tx.Commit(ctx)
}

func backfillOnce(
	ctx context.Context,
	pool *pgxpool.Pool,
	schema string,
	repo *tasks.Repo,
	rt *runtime.Runtime,
	lexicalSet map[string]struct{},
	semanticSet map[string]struct{},
	languages []string,
	list ListEntityIDsPage,
	pageSize int,
	maxPages int,
) error {
	if maxPages <= 0 || pageSize <= 0 {
		return nil
	}
	qs, err := pg.QuoteSchema(schema)
	if err != nil {
		return err
	}
	activeModels := rt.ActiveModels()
	pagesDone := 0

	// Lexical docs: fill missing documents.
	for et := range lexicalSet {
		for _, lang := range languages {
			if pagesDone >= maxPages {
				return nil
			}
			if strings.TrimSpace(lang) == "" {
				continue
			}

			cursor, state, err := ensureAndGetDocBackfillState(ctx, pool, qs, et, lang)
			if err != nil {
				return err
			}
			if state == "done" {
				continue
			}

			ids, nextCursor, done, err := list(ctx, et, lang, cursor, pageSize)
			if err != nil {
				_, _ = pool.Exec(ctx, fmt.Sprintf(`
					UPDATE %s.search_documents_backfill_state
					SET last_error = $3, state = 'failed', updated_at = now()
					WHERE entity_type = $1 AND language = $2
				`, qs), et, lang, err.Error())
				return err
			}
			if len(ids) > 0 {
				docs, err := rt.BuildLexicalString(ctx, et, lang, ids)
				if err != nil {
					return err
				}
				if err := pg.UpsertSearchDocuments(ctx, pool, schema, et, lang, docs); err != nil {
					return err
				}
			}
			if done {
				_, _ = pool.Exec(ctx, fmt.Sprintf(`
					UPDATE %s.search_documents_backfill_state
					SET cursor = $3, state = 'done', last_error = NULL, updated_at = now()
					WHERE entity_type = $1 AND language = $2
				`, qs), et, lang, nextCursor)
			} else {
				_, _ = pool.Exec(ctx, fmt.Sprintf(`
					UPDATE %s.search_documents_backfill_state
					SET cursor = $3, state = 'running', last_error = NULL, updated_at = now()
					WHERE entity_type = $1 AND language = $2
				`, qs), et, lang, nextCursor)
			}

			pagesDone++
		}
	}

	// Semantic: enqueue missing embeddings for active models.
	for et := range semanticSet {
		for _, lang := range languages {
			for _, model := range activeModels {
				if pagesDone >= maxPages {
					return nil
				}
				cursor, state, err := ensureAndGetVecBackfillState(ctx, pool, qs, model, et, lang)
				if err != nil {
					return err
				}
				if state == "done" {
					continue
				}
				ids, nextCursor, done, err := list(ctx, et, lang, cursor, pageSize)
				if err != nil {
					_, _ = pool.Exec(ctx, fmt.Sprintf(`
						UPDATE %s.embedding_vectors_backfill_state
						SET last_error = $4, state = 'failed', updated_at = now()
						WHERE model = $1 AND entity_type = $2 AND language = $3
					`, qs), model, et, lang, err.Error())
					return err
				}
				if len(ids) > 0 {
					missing, err := pg.FilterMissingEmbeddings(ctx, pool, schema, et, model, lang, ids)
					if err != nil {
						return err
					}
					if err := repo.EnqueueMany(ctx, et, missing, model, lang, "model_backfill"); err != nil {
						return err
					}
				}
				if done {
					_, _ = pool.Exec(ctx, fmt.Sprintf(`
						UPDATE %s.embedding_vectors_backfill_state
						SET cursor = $4, state = 'done', last_error = NULL, updated_at = now()
						WHERE model = $1 AND entity_type = $2 AND language = $3
					`, qs), model, et, lang, nextCursor)
				} else {
					_, _ = pool.Exec(ctx, fmt.Sprintf(`
						UPDATE %s.embedding_vectors_backfill_state
						SET cursor = $4, state = 'running', last_error = NULL, updated_at = now()
						WHERE model = $1 AND entity_type = $2 AND language = $3
					`, qs), model, et, lang, nextCursor)
				}
				pagesDone++
			}
		}
	}

	return nil
}

func ensureAndGetDocBackfillState(ctx context.Context, pool *pgxpool.Pool, qs string, entityType string, language string) (cursor string, state string, err error) {
	if _, err := pool.Exec(ctx, fmt.Sprintf(`
		INSERT INTO %s.search_documents_backfill_state (entity_type, language)
		VALUES ($1, $2)
		ON CONFLICT (entity_type, language) DO NOTHING
	`, qs), entityType, language); err != nil {
		return "", "", err
	}
	if err := pool.QueryRow(ctx, fmt.Sprintf(`
		SELECT cursor, state
		FROM %s.search_documents_backfill_state
		WHERE entity_type = $1 AND language = $2
	`, qs), entityType, language).Scan(&cursor, &state); err != nil {
		return "", "", err
	}
	return cursor, state, nil
}

func ensureAndGetVecBackfillState(ctx context.Context, pool *pgxpool.Pool, qs string, model string, entityType string, language string) (cursor string, state string, err error) {
	if _, err := pool.Exec(ctx, fmt.Sprintf(`
		INSERT INTO %s.embedding_vectors_backfill_state (model, entity_type, language)
		VALUES ($1, $2, $3)
		ON CONFLICT (model, entity_type, language) DO NOTHING
	`, qs), model, entityType, language); err != nil {
		return "", "", err
	}
	if err := pool.QueryRow(ctx, fmt.Sprintf(`
		SELECT cursor, state
		FROM %s.embedding_vectors_backfill_state
		WHERE model = $1 AND entity_type = $2 AND language = $3
	`, qs), model, entityType, language).Scan(&cursor, &state); err != nil {
		return "", "", err
	}
	return cursor, state, nil
}
