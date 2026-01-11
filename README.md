# embeddingkit

`embeddingkit` is a Go library for:

- generating embeddings (text, and optionally vision-language),
- storing/searching vectors (via app-provided storage),
- running background embedding jobs (task table + optional River worker helpers).

This library is intentionally **domain-agnostic**: the host application provides
callbacks for building text/documents and fetching assets, and an implementation
of the storage interface that matches its own Postgres schema (typically
`halfvec(K)` columns + HNSW indexes).

## Design goals

- Multiple embedding models without schema churn: store embeddings as **rows
  keyed by `(entity_type, entity_id, model)`**.
- `halfvec(K)` everywhere for ANN search (apps choose per-model tables/cols).
- Normalize vectors (L2) on write and on query.
- Two-stage retrieval ready: binary quantized oversample + fp16 rescoring.

## Non-goals / constraints

- No self-hosting guidance: embeddingkit assumes hosted embedding APIs.
- VL is URL-only: callers provide presigned/public URLs; embeddingkit does not upload raw bytes to providers.
- Job-system agnostic: River is supported as an optional adapter, not a requirement.

## What embeddingkit owns

- OpenAI-compatible text embedder client (DeepInfra/DashScope/etc).
- Interfaces for multimodal embedder(s) (VL).
- Task table migrations + repositories for enqueueing work.
- Task processing is job-system agnostic.
- Query helper utilities (pgvector/halfvec expression helpers).

## What the host app owns

- Domain-specific document construction (how to turn “gallery/video” into text).
- Asset selection + fetching (what images/frames to embed).
- Postgres storage schema for embeddings + indexes.
- Business rules and API response shapes.

## Postgres tables / migrations (River-style)

embeddingkit does **not** create its own schema. Like River, it creates prefixed
tables inside the host application schema via `search_path`.

Current tables:

- `embedding_tasks`
- `embedding_vectors`

## Migrations via migratekit (required)

embeddingkit migrations are intended to be applied and tracked with migratekit
(`public.migrations`), under `app=embeddingkit`.

This uses migratekit's schema targeting support (via `SET LOCAL search_path =
<host_schema>, public`).

Example (host app):

```go
import (
	"github.com/doujins-org/embeddingkit/migrations"
	"github.com/doujins-org/migratekit"
)

migs, _ := migratekit.LoadFromFS(migrations.Postgres)
m := migratekit.NewPostgres(sqlDB, "embeddingkit").WithSchema("doujins") // or "hentai0"
_ = m.ApplyMigrations(ctx, migs)
```

## Usage (host app)

### 1) Create an embedder

Use `embedder.NewOpenAICompatible(...)` with the provider’s OpenAI-compatible
base URL + API key + model name.

### 2) Run embeddingkit migrations

Host apps should apply embeddingkit migrations as part of their normal migration
flow using migratekit (see above). embeddingkit does not ship a standalone
migration runner.

### 3) Enqueue work and run workers

- Enqueue with `tasks.Repo.Enqueue(...)` or `runtime.Runtime.EnqueueTextEmbedding(...)`.
- Process tasks with either:
  - your own job runner that calls `tasks.Repo.FetchReady` + `runtime.Runtime.GenerateAndStoreEmbedding` + `tasks.Repo.Complete/Fail`.

The host app provides:

- `runtime.DocumentBuilder` (entity → text),
- `runtime.Storage` (upsert embedding vectors).

If you want a default Postgres implementation that writes to the embeddingkit
tables, use `pg.NewPostgresStorage(pool, schema)` (writes to
`<schema>.embedding_vectors`).

#### Example: non-River worker loop (optional)

```go
// package yourapp

import (
	"context"
	"time"

	"github.com/doujins-org/embeddingkit/runtime"
	"github.com/doujins-org/embeddingkit/tasks"
)

func RunEmbeddingWorker(ctx context.Context, rt *runtime.Runtime, repo *tasks.Repo) error {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			batch, err := repo.FetchReady(ctx, 250, 30*time.Second)
			if err != nil {
				return err
			}
			for _, task := range batch {
				if err := rt.GenerateAndStoreEmbedding(ctx, task.EntityType, task.EntityID, task.Model); err != nil {
					_ = repo.Fail(ctx, task.ID, 30*time.Second)
					continue
				}
				_ = repo.Complete(ctx, task.ID)
			}
		}
	}
}
```

## Vector search (candidate generation)

embeddingkit can generate semantic candidates from stored vectors in
`<schema>.embedding_vectors`.

- Query text → candidates: use `search.SearchVectors(...)` with the query vector.
- Similar-to-item → candidates: use `search.SimilarTo(...)` to find neighbors of an existing stored vector.

These APIs return only `(entity_type, entity_id, model, similarity)`; the host
app hydrates those IDs into domain rows and applies business rules.

## Post-processing

For diversity/reranking, embeddingkit provides a minimal Maximal Marginal
Relevance helper:

- `search.MMRReRank(...)`

The caller supplies a candidate-to-candidate similarity function (embeddingkit
does not assume it can fetch vectors for candidates).

## Evaluation

`eval` provides a tiny metrics skeleton for hand-written query sets:

- `eval.RecallAtK(...)`
- `eval.MRR(...)`

#### Example: River worker (optional)

`embeddingkit` does not depend on River, but if your app uses River already you can wire up a worker like:

```go
// package yourapp

import (
	"context"
	"time"

	"github.com/riverqueue/river"

	"github.com/doujins-org/embeddingkit/runtime"
	"github.com/doujins-org/embeddingkit/tasks"
)

type EmbeddingTaskBatchArgs struct {
	Limit int `json:"limit"`
}

func (EmbeddingTaskBatchArgs) Kind() string { return "embedding_tasks_batch" }

type EmbeddingTaskBatchWorker struct {
	river.WorkerDefaults[EmbeddingTaskBatchArgs]

	Runtime   *runtime.Runtime
	TaskRepo  *tasks.Repo
	LockAhead time.Duration
	Backoff   time.Duration
}

func (w *EmbeddingTaskBatchWorker) Work(ctx context.Context, job *river.Job[EmbeddingTaskBatchArgs]) error {
	limit := job.Args.Limit
	if limit <= 0 {
		limit = 250
	}

	lockAhead := w.LockAhead
	if lockAhead <= 0 {
		lockAhead = 30 * time.Second
	}

	backoff := w.Backoff
	if backoff <= 0 {
		backoff = 30 * time.Second
	}

	tasksToRun, err := w.TaskRepo.FetchReady(ctx, limit, lockAhead)
	if err != nil {
		return err
	}

	for _, task := range tasksToRun {
		if err := w.Runtime.GenerateAndStoreEmbedding(ctx, task.EntityType, task.EntityID, task.Model); err != nil {
			_ = w.TaskRepo.Fail(ctx, task.ID, backoff)
			continue
		}
		_ = w.TaskRepo.Complete(ctx, task.ID)
	}

	return nil
}
```

## VL embeddings (hosted-only; provider TBD)

Qwen/Qwen3-VL-Embedding-* is not currently available as a hosted API endpoint we
can depend on. embeddingkit still models VL support so it can plug into a future
hosted provider.

Reference script (pool-last-token + L2 normalize):
`https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/blob/main/scripts/qwen3_vl_embedding.py`

Planned contract:

- Input: (optional text) + N image/frame URLs (and optionally a single video URL) → ONE fused vector.
- URL-only: embeddingkit does not upload raw bytes/streams to providers.
- Limits: embeddingkit caps assets via `runtime.Config.MaxAssets` (default 8). Any provider-specific pixel/size/token limits are outside embeddingkit.
- Storage/search: embeddingkit stores VL vectors in the same canonical `embedding_vectors` table (keyed by `(entity_type, entity_id, model)`).

Interface:

- `vl.Embedder` is URL-only: `EmbedTextAndAssetURLs(ctx, text, []vl.AssetURL) -> []float32`.

Chunking:

- If you need to embed more than `MaxAssets`, callers can chunk externally and fuse chunk vectors with `vl.FuseAverageL2(...)`.

Models:

- embeddingkit treats `model` as an opaque string stored in `embedding_vectors.model`.
- Switching the “active model” is app-level: pick a model name and pass it into `runtime.GenerateAndStoreEmbedding(...)` and `search.SearchVectors(...)`.

Indexes:

- embeddingkit ships a general HNSW cosine index on `embedding_vectors.embedding`.
- If you want two-stage retrieval (binary quantize oversample + rescore), add an additional expression index in the host app migrations.

## Rollout / migrations (recommended)

- Dual-write embeddings for old+new models during rollout (enqueue tasks for both).
- Switch the “active search model” (app-level view/index) only after backfill.
- Keep search stable by:
  - using per-model partial indexes (WHERE model=...),
  - rolling model changes behind config flags,
  - and keeping a fallback path (lexical/trigram or previous model) if the new model is missing coverage.
