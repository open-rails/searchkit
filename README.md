# embeddingkit

`embeddingkit` is a Go library for generating embeddings, storing/searching
vectors in Postgres (pgvector), and running background embedding jobs via a
simple task table.

This README is a **manual** for host applications. For design notes and deeper
details, see `agents/NOTES.md`.

## Host app integration (manual)

### 1) Apply Postgres migrations (required)

embeddingkit migrations are intended to be applied and tracked with migratekit
(`public.migrations`), under `app=embeddingkit`.

This uses migratekit's schema targeting support (via `SET LOCAL search_path =
<host_schema>, public`).

Example (host app):

```go
import (
	"context"
	"database/sql"

	"github.com/doujins-org/embeddingkit/migrations"
	"github.com/doujins-org/migratekit"
)

func applyEmbeddingkitMigrations(ctx context.Context, sqlDB *sql.DB, schema string) error {
	migs, err := migratekit.LoadFromFS(migrations.Postgres)
	if err != nil {
		return err
	}
	m := migratekit.NewPostgres(sqlDB, "embeddingkit").WithSchema(schema)
	if err := m.ApplyMigrations(ctx, migs); err != nil {
		return err
	}
	return m.ValidateAllApplied(ctx, migs)
}
```

### 2) Create embedders (text, and optionally VL)

Use `embedder.NewOpenAICompatible(...)` with your provider’s OpenAI-compatible
base URL + API key + model name.

For VL, the contract is URL-only (the host app provides presigned/public URLs).

### 3) Wire host callbacks

Host apps provide:

- `runtime.BuildText`: `(entity_type, []entity_id) -> map[entity_id]text` (batch-first)
- `vl.ListAssetURLs`: `(entity_type, []entity_id) -> map[entity_id][]assets` (URL-only; batch-first)

### 4) Enqueue work

When domain entities change (or when you want to backfill), enqueue tasks via:

- `tasks.Repo.Enqueue(ctx, entityType, entityID, model, reason)`

Deduplication is by `(entity_type, entity_id, model)`.

### 5) Run workers

You can run workers with any job runner you want. A minimal loop is:

- `tasks.Repo.FetchReady(...)`
- `runtime.Runtime.GenerateAndStoreEmbedding(...)` (or batch via `embedder.EmbedTexts(...)`)
- `tasks.Repo.Complete(...)` / `tasks.Repo.Fail(...)` / `tasks.Repo.DeadLetter(...)`

Or use the optional helper:

- `worker.Run(ctx, rt, repo, worker.Options{...})` / `worker.DrainOnce(...)`
  - Provider calls are batched for text embeddings (25 texts per request).

Example (non-River, using embeddingkit worker helper):

```go
// package yourapp

import (
	"context"
	"time"

	"github.com/doujins-org/embeddingkit/runtime"
	"github.com/doujins-org/embeddingkit/tasks"
	"github.com/doujins-org/embeddingkit/worker"
)

func RunEmbeddingWorker(ctx context.Context, rt *runtime.Runtime, repo *tasks.Repo) error {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			_ = worker.DrainOnce(ctx, rt, repo, worker.Options{})
		}
	}
}
```

### 6) Query candidates (vector search)

embeddingkit can generate semantic candidates from stored vectors in
`<schema>.embedding_vectors`.

- Query text → candidates: use `search.SearchVectors(...)` with the query vector.
- Similar-to-item → candidates: use `search.SimilarTo(...)` to find neighbors of an existing stored vector.

These APIs return only `(entity_type, entity_id, model, similarity)`; the host
app hydrates those IDs into domain rows and applies business rules.

#### App-owned filtering inside KNN (e.g. language availability)

If the host app needs to enforce additional constraints *inside* the vector
query (for example, “only return galleries that have a translation row for the
requested language”), pass an additional WHERE fragment via:

- `search.Options.FilterSQL`
- `search.Options.FilterArgs` (named args referenced as `@name`)

This fragment is appended as:

`... AND (<FilterSQL>)`

`FilterSQL` is **trusted SQL owned by the host app**. Do not interpolate user
input into it unsafely.

## Model registry + indexes (recommended)

If you want embeddingkit to be configuration-driven (and to ensure per-model ANN
indexes exist automatically), construct the runtime via:

- `runtime.NewWithContext(ctx, runtime.Options{...})`

This will upsert configured models into `<schema>.embedding_models` and create
per-model cosine + binary HNSW indexes (using `CREATE INDEX CONCURRENTLY`).

If you also provide `BackfillEntityTypes` + `ListEntityIDsPage` in
`runtime.Options`, embeddingkit will start a background loop that enqueues
`embedding_tasks` for model backfills in bounded batches.
