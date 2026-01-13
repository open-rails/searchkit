-- searchkit: Postgres schema for lexical + semantic search (single migration).
--
-- IMPORTANT: this intentionally follows River's pattern: create tables in the
-- host application's schema (no separate searchkit schema).
--
-- searchkit is config-driven at runtime:
--   - lexical docs are stored in search_documents (pg_trgm)
--   - semantic vectors are stored in embedding_vectors (pgvector/halfvec)
--   - models are registered in embedding_models
--   - per-(model, language) cosine+binary ANN indexes are created CONCURRENTLY
--     at runtime (so we do not ship a global ANN index here)

-- pg_trgm provides trigram similarity operators + GIN index support.
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- pgvector provides the vector/halfvec types + HNSW indexes + operators.
CREATE EXTENSION IF NOT EXISTS vector;

-- ----------------------------------------------------------------------------
-- Lexical document store (trigram/typeahead).
-- searchkit heavy-normalizes before storing.
-- ----------------------------------------------------------------------------
CREATE TABLE search_documents (
    entity_type text NOT NULL,
    entity_id text NOT NULL,
    language text NOT NULL,
    document text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entity_type, entity_id, language)
);

CREATE INDEX idx_search_documents_entity_language
    ON search_documents(entity_type, language);

CREATE INDEX idx_search_documents_document_gin
    ON search_documents USING gin (document gin_trgm_ops);

-- ----------------------------------------------------------------------------
-- Dirty queue: host marks (entity_type, entity_id, language) as changed.
-- searchkit decides what to rebuild based on runtime config.
-- ----------------------------------------------------------------------------
CREATE TABLE search_dirty (
    entity_type text NOT NULL,
    entity_id text NOT NULL,
    language text NOT NULL,
    is_deleted boolean NOT NULL DEFAULT false,
    reason text NOT NULL DEFAULT 'unknown',
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entity_type, entity_id, language)
);

CREATE INDEX idx_search_dirty_updated_at
    ON search_dirty(updated_at);

-- ----------------------------------------------------------------------------
-- Task queue (one task per entity+model; PK is the natural identity).
-- ----------------------------------------------------------------------------
CREATE TABLE embedding_tasks (
    entity_type text NOT NULL,
    entity_id text NOT NULL,
    model text NOT NULL,
    language text NOT NULL,
    reason text NOT NULL DEFAULT 'unknown',
    attempts integer NOT NULL DEFAULT 0,
    next_run_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at timestamptz NULL,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entity_type, entity_id, model, language)
);

-- Leasing/ready scan index.
CREATE INDEX idx_embedding_tasks_ready
    ON embedding_tasks(next_run_at, entity_type, entity_id, model, language);

-- ----------------------------------------------------------------------------
-- Canonical embedding store (one vector per entity+model).
-- ----------------------------------------------------------------------------
CREATE TABLE embedding_vectors (
    entity_type text NOT NULL,
    entity_id text NOT NULL,
    model text NOT NULL,
    language text NOT NULL,
    embedding halfvec,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entity_type, entity_id, model, language)
);

CREATE INDEX idx_embedding_vectors_model
    ON embedding_vectors(model, language);

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
CREATE TABLE embedding_vectors_backfill_state (
    model text NOT NULL,
    entity_type text NOT NULL,
    language text NOT NULL,
    cursor text NOT NULL DEFAULT '',
    state text NOT NULL DEFAULT 'running', -- running|done|failed
    last_error text,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model, entity_type, language)
);

CREATE INDEX idx_embedding_vectors_backfill_state_state
    ON embedding_vectors_backfill_state(state);

-- Lexical backfill state (cursor-driven initial fill).
CREATE TABLE search_documents_backfill_state (
    entity_type text NOT NULL,
    language text NOT NULL,
    cursor text NOT NULL DEFAULT '',
    state text NOT NULL DEFAULT 'running', -- running|done|failed
    last_error text,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entity_type, language)
);

CREATE INDEX idx_search_documents_backfill_state_state
    ON search_documents_backfill_state(state);

-- ----------------------------------------------------------------------------
-- Dead-letter queue (terminal failures only; keeps embedding_tasks mostly empty).
-- ----------------------------------------------------------------------------
CREATE TABLE embedding_dead_letters (
    entity_type text NOT NULL,
    entity_id text NOT NULL,
    model text NOT NULL,
    language text NOT NULL,
    reason text NOT NULL,
    error text NOT NULL,
    attempts integer NOT NULL,
    failed_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entity_type, entity_id, model, language)
);

CREATE INDEX idx_embedding_dead_letters_failed_at
    ON embedding_dead_letters(failed_at);
