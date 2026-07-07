-- searchkit: Postgres schema for lexical + semantic search — consolidated
-- baseline (squashed from 001..003, 2026-07-07).
--
-- IMPORTANT: this intentionally follows River's pattern: create tables in the
-- host application's schema (no separate searchkit schema). Migrations are
-- unqualified and applied with search_path scoped to the host schema.
--
-- searchkit is config-driven at runtime:
--   - lexical docs are stored in search_documents (pg_trgm + FTS + PGroonga)
--   - semantic vectors are stored in embedding_vectors (pgvector/halfvec)
--   - models are registered in embedding_models
--   - per-(model, language) cosine+binary ANN indexes are created CONCURRENTLY
--     at runtime (so we do not ship a global ANN index here)
--
-- migratekit tracks applied migrations by filename, so existing databases
-- skip this file. New migrations start at 0002.

-- ----------------------------------------------------------------------------
-- Extensions.
-- ----------------------------------------------------------------------------

-- pg_trgm provides trigram similarity operators + GIN index support.
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- pgvector provides the vector/halfvec types + HNSW indexes + operators.
CREATE EXTENSION IF NOT EXISTS vector;

-- pgroonga backs native-script (CJK/Korean) full-text matching. Requires
-- superuser or elevated privileges; if your environment can't run CREATE
-- EXTENSION from app migrations, install/enable it out-of-band and mark this
-- migration applied.
CREATE EXTENSION IF NOT EXISTS pgroonga;

-- ----------------------------------------------------------------------------
-- Functions.
-- ----------------------------------------------------------------------------

-- Map a BCP-47-ish language code (e.g. "en", "es") to a Postgres regconfig.
-- Most installations only ship a subset of configs; we default to `simple`.
CREATE OR REPLACE FUNCTION searchkit_regconfig_for_language(lang text)
RETURNS regconfig
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT CASE lower(trim(coalesce(lang, '')))
        WHEN 'en' THEN 'english'::regconfig
        WHEN 'es' THEN 'spanish'::regconfig
        WHEN 'fr' THEN 'french'::regconfig
        WHEN 'de' THEN 'german'::regconfig
        WHEN 'it' THEN 'italian'::regconfig
        WHEN 'pt' THEN 'portuguese'::regconfig
        WHEN 'ru' THEN 'russian'::regconfig
        -- For languages without a built-in stemmer config (e.g. ja/ko/zh),
        -- `simple` still tokenizes reasonably and is deterministic.
        ELSE 'simple'::regconfig
    END;
$$;

-- ----------------------------------------------------------------------------
-- Lexical document store (typeahead + FTS + native-script search).
-- searchkit heavy-normalizes `document` before storing; `raw_document` keeps
-- the host-provided text for FTS/PGroonga, which prefer it over `document`.
-- ----------------------------------------------------------------------------
CREATE TABLE search_documents (
    entity_type text NOT NULL,
    entity_id text NOT NULL,
    language text NOT NULL,
    document text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    raw_document text,
    tsv tsvector,
    PRIMARY KEY (entity_type, entity_id, language)
);

CREATE INDEX idx_search_documents_entity_language
    ON search_documents(entity_type, language);

-- Trigram index for typeahead.
CREATE INDEX idx_search_documents_document_gin
    ON search_documents USING gin (document gin_trgm_ops);

-- FTS index (BM25-family lexical search).
CREATE INDEX idx_search_documents_tsv_gin
    ON search_documents USING gin (tsv);

-- PGroonga full-text index for native-script queries (primary for ja/zh/ko).
-- Partial index keeps size manageable while targeting languages that most
-- need segmentation.
CREATE INDEX idx_search_documents_raw_document_pgroonga_cjk
    ON search_documents
 USING pgroonga (raw_document)
 WHERE language IN ('ja', 'zh', 'ko');

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
