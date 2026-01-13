-- searchkit: add Postgres FTS (BM25-family) support for lexical search.
--
-- This migration extends `search_documents` so the same host-provided lexical
-- document can power:
--   - trigram/typeahead (heavy-normalized `document` + gin_trgm_ops)
--   - full-text search (raw-ish `raw_document` + `tsv` GIN index)
--
-- NOTE: This does not change the host callback interface. Hosts still provide
-- one lexical document per (entity_type, entity_id, language); searchkit stores
-- both representations internally.

BEGIN;

ALTER TABLE search_documents
    ADD COLUMN IF NOT EXISTS raw_document text,
    ADD COLUMN IF NOT EXISTS tsv tsvector;

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

-- Backfill existing rows (old schema stored only the heavy-normalized document).
UPDATE search_documents
SET raw_document = document
WHERE raw_document IS NULL;

UPDATE search_documents
SET tsv = to_tsvector(searchkit_regconfig_for_language(language), coalesce(raw_document, ''))
WHERE tsv IS NULL;

-- FTS index.
CREATE INDEX IF NOT EXISTS idx_search_documents_tsv_gin
    ON search_documents USING gin (tsv);

COMMIT;

