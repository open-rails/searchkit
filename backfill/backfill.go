package backfill

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/doujins-org/embeddingkit/pg"
	"github.com/doujins-org/embeddingkit/tasks"
)

// ListEntityIDsPage returns a page of entity IDs for a given entity type.
//
// cursor is an opaque string (interpreted only by the host app).
// nextCursor is the cursor to resume from on the next page.
// done indicates there are no more entities after this page.
type ListEntityIDsPage func(ctx context.Context, entityType string, cursor string, limit int) (ids []string, nextCursor string, done bool, err error)

type Options struct {
	// Defaults are chosen to be "fast but safe" without overwhelming providers.
	PageSize       int
	MaxTasksPerRun int
	MaxRuntime     time.Duration
}

func (o *Options) withDefaults() Options {
	out := Options{
		PageSize:       o.PageSize,
		MaxTasksPerRun: o.MaxTasksPerRun,
		MaxRuntime:     o.MaxRuntime,
	}
	if out.PageSize <= 0 {
		out.PageSize = 1000
	}
	if out.MaxTasksPerRun <= 0 {
		out.MaxTasksPerRun = 50_000
	}
	if out.MaxRuntime <= 0 {
		out.MaxRuntime = 30 * time.Second
	}
	return out
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

// RunOnce performs a bounded amount of backfill work for the given models and entity types.
//
// This is designed to be called periodically (e.g. in a background loop) so large
// backfills (millions of entities) don't block startup.
func RunOnce(ctx context.Context, pool *pgxpool.Pool, schema string, repo *tasks.Repo, models []pg.ModelSpec, entityTypes []string, list ListEntityIDsPage, opts Options) (int, error) {
	if pool == nil {
		return 0, fmt.Errorf("pool is required")
	}
	if repo == nil {
		return 0, fmt.Errorf("task repo is required")
	}
	if strings.TrimSpace(schema) == "" {
		return 0, fmt.Errorf("schema is required")
	}
	if list == nil {
		return 0, fmt.Errorf("ListEntityIDsPage is required")
	}
	if len(models) == 0 || len(entityTypes) == 0 {
		return 0, nil
	}

	cfg := opts.withDefaults()
	start := time.Now()

	qs, err := quoteIdent(schema)
	if err != nil {
		return 0, fmt.Errorf("invalid schema: %w", err)
	}

	enqueued := 0

	// Loop models x entity types, spending a bounded budget per run.
	for _, m := range models {
		model := strings.TrimSpace(m.Name)
		if model == "" {
			continue
		}
		for _, entityType := range entityTypes {
			if time.Since(start) > cfg.MaxRuntime || enqueued >= cfg.MaxTasksPerRun {
				return enqueued, nil
			}
			et := strings.TrimSpace(entityType)
			if et == "" {
				continue
			}

			// Ensure state row exists.
			_, _ = pool.Exec(ctx, fmt.Sprintf(`
				INSERT INTO %s.embedding_backfill_state (model, entity_type, cursor, state, updated_at)
				VALUES ($1, $2, '', 'running', now())
				ON CONFLICT (model, entity_type) DO NOTHING
			`, qs), model, et)

			// Load cursor/state.
			var cursor string
			var state string
			if err := pool.QueryRow(ctx, fmt.Sprintf(`
				SELECT cursor, state
				FROM %s.embedding_backfill_state
				WHERE model = $1 AND entity_type = $2
				LIMIT 1
			`, qs), model, et).Scan(&cursor, &state); err != nil {
				return enqueued, err
			}
			if state == "done" {
				continue
			}

			ids, nextCursor, done, err := list(ctx, et, cursor, cfg.PageSize)
			if err != nil {
				_, _ = pool.Exec(ctx, fmt.Sprintf(`
					UPDATE %s.embedding_backfill_state
					SET last_error = $3, updated_at = now()
					WHERE model = $1 AND entity_type = $2
				`, qs), model, et, err.Error())
				return enqueued, err
			}

			for _, id := range ids {
				if time.Since(start) > cfg.MaxRuntime || enqueued >= cfg.MaxTasksPerRun {
					break
				}
				if strings.TrimSpace(id) == "" {
					continue
				}
				if err := repo.Enqueue(ctx, et, id, model, "model_backfill"); err != nil {
					return enqueued, err
				}
				enqueued++
			}

			// Advance cursor and/or mark done.
			if done {
				_, _ = pool.Exec(ctx, fmt.Sprintf(`
					UPDATE %s.embedding_backfill_state
					SET cursor = $3, state = 'done', last_error = NULL, updated_at = now()
					WHERE model = $1 AND entity_type = $2
				`, qs), model, et, nextCursor)
			} else {
				_, _ = pool.Exec(ctx, fmt.Sprintf(`
					UPDATE %s.embedding_backfill_state
					SET cursor = $3, last_error = NULL, updated_at = now()
					WHERE model = $1 AND entity_type = $2
				`, qs), model, et, nextCursor)
			}
		}
	}

	return enqueued, nil
}
