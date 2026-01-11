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

## What embeddingkit owns

- OpenAI-compatible text embedder client (DeepInfra/DashScope/etc).
- Interfaces for multimodal embedder(s) (VL).
- Task table migrations + repositories for enqueueing work.
- Optional River worker helpers for processing tasks (apps may use any job system).
- Query helper utilities (pgvector/halfvec expression helpers).

## What the host app owns

- Domain-specific document construction (how to turn “gallery/video” into text).
- Asset selection + fetching (what images/frames to embed).
- Postgres storage schema for embeddings + indexes.
- Business rules and API response shapes.

## Usage (host app)

### 1) Create an embedder

Use `embedder.NewOpenAICompatible(...)` with the provider’s OpenAI-compatible
base URL + API key + model name.

### 2) Run embeddingkit migrations

`embeddingkit` currently ships only its task table, created in the host app
schema (River-style prefix):

- `embeddingkit_embedding_tasks`

Host apps can apply these migrations during startup/migration flow:

```go
_ = migrate.ApplyPostgres(ctx, pgxPool, "doujins") // or "hentai0", etc.
```

### 3) Enqueue work and run workers

- Enqueue with `tasks.Repo.Enqueue(...)` or `runtime.Runtime.EnqueueTextEmbedding(...)`.
- Process tasks with either:
  - `river.TaskBatchWorker` (if you use River), or
  - your own job runner that calls `tasks.Repo.FetchReady` + `runtime.Runtime.GenerateAndStoreEmbedding` + `tasks.Repo.Complete/Fail`.

The host app provides:

- `runtime.DocumentBuilder` (entity → text),
- `runtime.Storage` (upsert embedding into the app’s schema).

## Rollout / migrations (recommended)

- Dual-write embeddings for old+new models during rollout (enqueue tasks for both).
- Switch the “active search model” (app-level view/index) only after backfill.
- Keep search stable by:
  - using per-model partial indexes (WHERE model=...),
  - rolling model changes behind config flags,
  - and keeping a fallback path (lexical/trigram or previous model) if the new model is missing coverage.
