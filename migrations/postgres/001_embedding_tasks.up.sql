-- embeddingkit: Postgres schema for embeddings + task queue.
-- IMPORTANT: this intentionally follows River's pattern: create tables in the
-- host application's schema (no separate embeddingkit schema).

-- pgvector provides the vector/halfvec types + HNSW indexes + operators.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS embedding_tasks (
    id bigserial PRIMARY KEY,
    entity_type text NOT NULL,
    entity_id text NOT NULL,
    model text NOT NULL,
    reason text NOT NULL DEFAULT 'unknown',
    attempts integer NOT NULL DEFAULT 0,
    next_run_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_embedding_tasks_entity_model
    ON embedding_tasks(entity_type, entity_id, model);

CREATE INDEX IF NOT EXISTS idx_embedding_tasks_ready
    ON embedding_tasks(next_run_at, id);

-- Stores ONE embedding vector per (entity_type, entity_id, model).
-- Note: we intentionally use `halfvec` without a fixed dimension here so the
-- table can store multiple models/dims. Apps should still create model-specific
-- indexes if they want per-model tuning/partial indexes.
CREATE TABLE IF NOT EXISTS embedding_vectors (
    id bigserial PRIMARY KEY,
    entity_type text NOT NULL,
    entity_id text NOT NULL,
    model text NOT NULL,
    embedding halfvec,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_embedding_vectors_entity_model
    ON embedding_vectors(entity_type, entity_id, model);

CREATE INDEX IF NOT EXISTS idx_embedding_vectors_model
    ON embedding_vectors(model);

-- Generic ANN index (cosine distance). Host apps may add more specialized
-- partial indexes per model if desired.
CREATE INDEX IF NOT EXISTS idx_embedding_vectors_hnsw_cosine
    ON embedding_vectors USING hnsw (embedding halfvec_cosine_ops)
    WHERE embedding IS NOT NULL;
