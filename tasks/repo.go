package tasks

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

type Repo struct {
	pool   *pgxpool.Pool
	schema string
}

const embeddingTasksTable = "embedding_tasks"

func NewRepo(pool *pgxpool.Pool, schema string) *Repo {
	return &Repo{pool: pool, schema: schema}
}

func (r *Repo) Enqueue(ctx context.Context, entityType string, entityID string, model string, reason string) error {
	if entityType == "" || model == "" {
		return fmt.Errorf("entityType and model are required")
	}
	if strings.TrimSpace(entityID) == "" {
		return fmt.Errorf("entityID is required")
	}
	if r.schema == "" {
		return fmt.Errorf("schema is required")
	}
	q := fmt.Sprintf(`
		INSERT INTO %s.%s (entity_type, entity_id, model, reason)
		VALUES ($1, $2, $3, COALESCE($4, 'unknown'))
		ON CONFLICT (entity_type, entity_id, model) DO UPDATE SET
			reason = EXCLUDED.reason,
			next_run_at = LEAST(%s.%s.next_run_at, now()),
			updated_at = now()
	`, r.schema, embeddingTasksTable, r.schema, embeddingTasksTable)
	_, err := r.pool.Exec(ctx, q, entityType, entityID, model, reason)
	return err
}

// FetchReady returns up to limit tasks ready to run now, and bumps next_run_at
// forward by lockAhead to reduce duplicate work across workers.
func (r *Repo) FetchReady(ctx context.Context, limit int, lockAhead time.Duration) ([]Task, error) {
	if limit <= 0 {
		return nil, nil
	}
	if lockAhead <= 0 {
		lockAhead = 30 * time.Second
	}
	if r.schema == "" {
		return nil, fmt.Errorf("schema is required")
	}

	now := time.Now().UTC()
	next := now.Add(lockAhead)

	q := fmt.Sprintf(`
		WITH picked AS (
			SELECT id
			FROM %s.%s
			WHERE next_run_at <= $1
			ORDER BY next_run_at ASC, id ASC
			LIMIT $2
			FOR UPDATE SKIP LOCKED
		)
		UPDATE %s.%s t
		SET next_run_at = $3, updated_at = $1
		WHERE t.id IN (SELECT id FROM picked)
		RETURNING
			t.id, t.entity_type, t.entity_id, t.model, t.reason, t.attempts, t.next_run_at, t.created_at, t.updated_at
	`, r.schema, embeddingTasksTable, r.schema, embeddingTasksTable)

	rows, err := r.pool.Query(ctx, q, now, limit, next)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []Task
	for rows.Next() {
		var t Task
		if err := rows.Scan(
			&t.ID,
			&t.EntityType,
			&t.EntityID,
			&t.Model,
			&t.Reason,
			&t.Attempts,
			&t.NextRunAt,
			&t.CreatedAt,
			&t.UpdatedAt,
		); err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, rows.Err()
}

func (r *Repo) Complete(ctx context.Context, id int64) error {
	if id <= 0 {
		return nil
	}
	if r.schema == "" {
		return fmt.Errorf("schema is required")
	}
	q := fmt.Sprintf("DELETE FROM %s.%s WHERE id = $1", r.schema, embeddingTasksTable)
	_, err := r.pool.Exec(ctx, q, id)
	return err
}

func (r *Repo) Fail(ctx context.Context, id int64, backoff time.Duration) error {
	if id <= 0 {
		return nil
	}
	if backoff <= 0 {
		backoff = 30 * time.Second
	}
	if r.schema == "" {
		return fmt.Errorf("schema is required")
	}
	secs := int64(backoff / time.Second)
	if secs < 1 {
		secs = 1
	}
	q := fmt.Sprintf(`
		UPDATE %s.%s
		SET attempts = attempts + 1,
		    next_run_at = now() + make_interval(secs => $1),
		    updated_at = now()
		WHERE id = $2
	`, r.schema, embeddingTasksTable)
	_, err := r.pool.Exec(ctx, q, secs, id)
	return err
}
