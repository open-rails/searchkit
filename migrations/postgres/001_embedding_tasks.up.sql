-- embeddingkit: Postgres schema for embeddings + task queue (single migration).
--
-- IMPORTANT: this intentionally follows River's pattern: create tables in the
-- host application's schema (no separate embeddingkit schema).
--
-- embeddingkit is config-driven at runtime:
--   - models are registered in embedding_models
--   - per-model cosine+binary ANN indexes are created CONCURRENTLY at runtime
--     (so we do not ship a global ANN index here)

-- pgvector provides the vector/halfvec types + HNSW indexes + operators.
CREATE EXTENSION IF NOT EXISTS vector;

-- ----------------------------------------------------------------------------
-- Task queue (one task per entity+model; PK is the natural identity).
-- ----------------------------------------------------------------------------
CREATE TABLE embedding_tasks (
    entity_type text NOT NULL,
    entity_id text NOT NULL,
    model text NOT NULL,
    reason text NOT NULL DEFAULT 'unknown',
    attempts integer NOT NULL DEFAULT 0,
    next_run_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at timestamptz NULL,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entity_type, entity_id, model)
);

-- Leasing/ready scan index.
CREATE INDEX idx_embedding_tasks_ready
    ON embedding_tasks(next_run_at, entity_type, entity_id, model);

-- ----------------------------------------------------------------------------
-- Canonical embedding store (one vector per entity+model).
-- ----------------------------------------------------------------------------
CREATE TABLE embedding_vectors (
    entity_type text NOT NULL,
    entity_id text NOT NULL,
    model text NOT NULL,
    embedding halfvec,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entity_type, entity_id, model)
);

CREATE INDEX idx_embedding_vectors_model
    ON embedding_vectors(model);

-- ----------------------------------------------------------------------------
-- Model registry (synced from host config at runtime).
-- ----------------------------------------------------------------------------
CREATE TABLE embedding_models (
    model text PRIMARY KEY,
    dims integer NOT NULL CHECK (dims > 0),
    modality text NOT NULL, -- "text" | "vl" (future)
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_embedding_models_modality
    ON embedding_models(modality);

-- ----------------------------------------------------------------------------
-- Backfill state (enqueues tasks for newly-enabled models; opaque cursor).
-- ----------------------------------------------------------------------------
CREATE TABLE embedding_backfill_state (
    model text NOT NULL,
    entity_type text NOT NULL,
    cursor text NOT NULL DEFAULT '',
    state text NOT NULL DEFAULT 'running', -- running|done|failed
    last_error text,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model, entity_type)
);

CREATE INDEX idx_embedding_backfill_state_state
    ON embedding_backfill_state(state);

-- ----------------------------------------------------------------------------
-- Dead-letter queue (terminal failures only; keeps embedding_tasks mostly empty).
-- ----------------------------------------------------------------------------
CREATE TABLE embedding_dead_letters (
    entity_type text NOT NULL,
    entity_id text NOT NULL,
    model text NOT NULL,
    reason text NOT NULL,
    error text NOT NULL,
    attempts integer NOT NULL,
    failed_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entity_type, entity_id, model)
);

CREATE INDEX idx_embedding_dead_letters_failed_at
    ON embedding_dead_letters(failed_at);
