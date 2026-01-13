# searchkit

`searchkit` is a Go library for:

- **Typeahead / fuzzy lexical search** (language-specific) via Postgres `pg_trgm` over `search_documents.document`.
- **Keyword lexical search** (BM25-family; language-specific) via Postgres full-text search over `search_documents.tsv`.
- **Semantic search** (language-specific embeddings) via pgvector `halfvec` stored in `embedding_vectors`.
- A **single, host-run worker loop** that:
  - consumes `search_dirty` notifications (changed/deleted entities),
  - runs resumable cursor-based backfill (no “insert 10M dirty rows”),
  - and drains `embedding_tasks` to compute/store embeddings.

This README is a **manual** for host applications. Design notes live in `agents/NOTES.md`.

## Host app integration (manual)

### 1) Apply Postgres migrations (required)

searchkit migrations are applied/tracked with migratekit (`public.migrations`) under `app=searchkit`,
and are scoped to the host schema via `SET LOCAL search_path = <schema>, public`.

```go
import (
	"context"
	"database/sql"

	"github.com/doujins-org/migratekit"
	"github.com/doujins-org/searchkit/migrations"
)

func applySearchkitMigrations(ctx context.Context, sqlDB *sql.DB, schema string) error {
	migs, err := migratekit.LoadFromFS(migrations.Postgres)
	if err != nil {
		return err
	}
	m := migratekit.NewPostgres(sqlDB, "searchkit").WithSchema(schema)
	if err := m.ApplyMigrations(ctx, migs); err != nil {
		return err
	}
	return m.ValidateAllApplied(ctx, migs)
}
```

### 2) Create embedders (text, and optionally VL)

Use `embedder.NewOpenAICompatible(...)` with your provider’s OpenAI-compatible base URL + API key + model name.

For VL, the contract is URL-only (the host app provides presigned/public URLs).

### 3) Wire host callbacks (batch-first)

Host apps provide:

- `runtime.BuildSemanticDocument(ctx, entity_type, language, []entity_id) -> map[id]string` (**required**)
  - Used to generate embeddings.
- `runtime.BuildLexicalString(ctx, entity_type, language, []entity_id) -> map[id]string` (required if you want lexical docs)
  - Used to populate `search_documents` for both trigram typeahead and FTS.
- `vl.ListAssetURLs(ctx, entity_type, []entity_id) -> map[id][]AssetURL` (required only if VL models are enabled)

### 4) Mark changes (host writes `search_dirty`)

The host does **not** enqueue per-model tasks directly.
Instead, it upserts into `<schema>.search_dirty`:

- `(entity_type, entity_id, language, is_deleted, reason, updated_at)`

searchkit decides what to rebuild based on worker config + active model set.

### 5) Run one worker loop (host-owned, searchkit-provided)

Run a background worker (River/cron/goroutine) that calls:

- `worker.RunOnceSearchkit(ctx, rt, worker.SearchkitOptions{...})`

This single entrypoint:

1) processes `search_dirty`,
2) runs bounded backfill for missing docs/embeddings,
3) drains `embedding_tasks` (does provider calls and writes `embedding_vectors`).

### 6) Query candidates (lexical + semantic)

Recommended entrypoints:

- Typeahead (trigram-only): `searchkit.Typeahead(...)`
- Search (FTS + vector fused by RRF): `searchkit.Search(...)`

Lower-level building blocks (if you need direct control):

Lexical (typeahead / typos / substring):

- `search.LexicalSearch(ctx, pool, query, search.LexicalOptions{Schema, Language, EntityTypes, Limit, MinSimilarity})`

Lexical (keyword search; BM25-family):

- `search.FTSSearch(ctx, pool, query, search.FTSOptions{Schema, Language, EntityTypes, Limit})`

Semantic (candidate generation; host hydrates IDs + applies business logic):

- `search.SemanticSearch(ctx, pool, queryVec, search.Options{Schema, Model, Language, EntityTypes, ...})`
- Similar-to-item: `search.SimilarTo(ctx, pool, schema, entityType, entityID, model, language, limit, opts)`

These APIs return only IDs + scores; the host app hydrates those IDs into DTOs and blends results as desired.

## Language → Postgres FTS config mapping

FTS uses a schema-local function created by migrations:

- `<schema>.searchkit_regconfig_for_language(language)`

It maps common codes like `en/es/fr/de/...` to built-in configs and falls back to `simple`.

## Model registry + ANN indexes

Construct the runtime via `runtime.NewWithContext(...)` to:

- upsert the configured model set into `<schema>.embedding_models`, and
- ensure per-model cosine + binary HNSW indexes exist (via `CREATE INDEX CONCURRENTLY`).
