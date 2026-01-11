-- embeddingkit: task table only (embedding storage is app-owned)
-- IMPORTANT: this intentionally follows River's pattern: create prefixed tables
-- in the host application's schema (no separate embeddingkit schema).

CREATE TABLE IF NOT EXISTS embeddingkit_embedding_tasks (
    id bigserial PRIMARY KEY,
    entity_type text NOT NULL,
    entity_id bigint NOT NULL,
    model text NOT NULL,
    reason text NOT NULL DEFAULT 'unknown',
    attempts integer NOT NULL DEFAULT 0,
    next_run_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_embeddingkit_embedding_tasks_entity_model
    ON embeddingkit_embedding_tasks(entity_type, entity_id, model);

CREATE INDEX IF NOT EXISTS idx_embeddingkit_embedding_tasks_ready
    ON embeddingkit_embedding_tasks(next_run_at, id);
