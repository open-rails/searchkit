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
const embeddingDeadLettersTable = "embedding_dead_letters"

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
			SELECT entity_type, entity_id, model
			FROM %s.%s
			WHERE next_run_at <= $1
			ORDER BY next_run_at ASC, entity_type ASC, entity_id ASC, model ASC
			LIMIT $2
			FOR UPDATE SKIP LOCKED
		)
		UPDATE %s.%s t
		SET next_run_at = $3,
		    started_at = COALESCE(t.started_at, $1),
		    updated_at = $1
		FROM picked p
		WHERE t.entity_type = p.entity_type
		  AND t.entity_id = p.entity_id
		  AND t.model = p.model
		RETURNING
			t.entity_type, t.entity_id, t.model, t.reason, t.attempts, t.next_run_at, t.started_at, t.created_at, t.updated_at
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
			&t.EntityType,
			&t.EntityID,
			&t.Model,
			&t.Reason,
			&t.Attempts,
			&t.NextRunAt,
			&t.StartedAt,
			&t.CreatedAt,
			&t.UpdatedAt,
		); err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, rows.Err()
}

func (r *Repo) Complete(ctx context.Context, entityType string, entityID string, model string, leaseUntil time.Time) error {
	if r.schema == "" {
		return fmt.Errorf("schema is required")
	}
	if strings.TrimSpace(entityType) == "" || strings.TrimSpace(entityID) == "" || strings.TrimSpace(model) == "" {
		return nil
	}
	q := fmt.Sprintf(`
		DELETE FROM %s.%s
		WHERE entity_type = $1 AND entity_id = $2 AND model = $3 AND next_run_at = $4
	`, r.schema, embeddingTasksTable)
	_, err := r.pool.Exec(ctx, q, entityType, entityID, model, leaseUntil.UTC())
	return err
}

func (r *Repo) Fail(ctx context.Context, entityType string, entityID string, model string, leaseUntil time.Time, backoff time.Duration) error {
	if backoff <= 0 {
		backoff = 30 * time.Second
	}
	if r.schema == "" {
		return fmt.Errorf("schema is required")
	}
	if strings.TrimSpace(entityType) == "" || strings.TrimSpace(entityID) == "" || strings.TrimSpace(model) == "" {
		return nil
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
		WHERE entity_type = $2 AND entity_id = $3 AND model = $4 AND next_run_at = $5
	`, r.schema, embeddingTasksTable)
	_, err := r.pool.Exec(ctx, q, secs, entityType, entityID, model, leaseUntil.UTC())
	return err
}

// DeadLetter moves a task into the dead-letter table and deletes it from
// embedding_tasks so the runnable queue stays small.
//
// This is lease-safe: the task is deleted only if next_run_at matches leaseUntil.
func (r *Repo) DeadLetter(ctx context.Context, t Task, leaseUntil time.Time, err error) error {
	if r.schema == "" {
		return fmt.Errorf("schema is required")
	}
	if strings.TrimSpace(t.EntityType) == "" || strings.TrimSpace(t.EntityID) == "" || strings.TrimSpace(t.Model) == "" {
		return nil
	}
	if err == nil {
		err = fmt.Errorf("unknown error")
	}

	tx, txErr := r.pool.Begin(ctx)
	if txErr != nil {
		return txErr
	}
	defer func() { _ = tx.Rollback(ctx) }()

	q1 := fmt.Sprintf(`
		INSERT INTO %s.%s (entity_type, entity_id, model, reason, error, attempts, failed_at, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, now(), now(), now())
		ON CONFLICT (entity_type, entity_id, model) DO UPDATE SET
			reason = EXCLUDED.reason,
			error = EXCLUDED.error,
			attempts = EXCLUDED.attempts,
			failed_at = EXCLUDED.failed_at,
			updated_at = now()
	`, r.schema, embeddingDeadLettersTable)
	attempts := t.Attempts
	if attempts < 0 {
		attempts = 0
	}
	if _, execErr := tx.Exec(ctx, q1, t.EntityType, t.EntityID, t.Model, t.Reason, err.Error(), attempts); execErr != nil {
		return execErr
	}

	q2 := fmt.Sprintf(`
		DELETE FROM %s.%s
		WHERE entity_type = $1 AND entity_id = $2 AND model = $3 AND next_run_at = $4
	`, r.schema, embeddingTasksTable)
	if _, execErr := tx.Exec(ctx, q2, t.EntityType, t.EntityID, t.Model, leaseUntil.UTC()); execErr != nil {
		return execErr
	}

	return tx.Commit(ctx)
}
