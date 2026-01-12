# embeddingkit notes

This document holds implementation notes and extra details that are intentionally
not part of the main `README.md` manual.

## Design goals

- Multiple embedding models without schema churn: store embeddings as rows keyed
  by `(entity_type, entity_id, model)`.
- Two-stage retrieval ready (planned): binary quantized oversample + fp16/halfvec
  rescoring.
- Hosted-only VL: URL-only inputs to providers (no raw bytes uploads).
- Job-runner agnostic: embeddingkit does not require River.

## Postgres tables

embeddingkit migrations create tables in the host schema (River-style via
`search_path`):

- `embedding_tasks`
- `embedding_vectors`

## VL embeddings (hosted-only; provider TBD)

Qwen/Qwen3-VL-Embedding-* is not currently available as a hosted API endpoint we
can depend on. embeddingkit models VL support so it can plug into a future
hosted provider.

Reference script (pool-last-token + L2 normalize):
`https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/blob/main/scripts/qwen3_vl_embedding.py`

Planned contract:

- Input: (optional text) + N image/frame URLs (and optionally a single video URL)
  → ONE fused vector.
- URL-only: embeddingkit does not upload raw bytes/streams to providers.
- Asset selection/chunking is host-app owned.

## Candidate generation

- `search.SearchVectors(...)` performs KNN candidate generation from stored
  vectors.
- `search.SimilarTo(...)` finds nearest neighbors of an existing stored vector.

These return only IDs + similarity. Host apps hydrate IDs into domain rows and
apply business rules.

### App-owned filtering inside KNN

Some apps need constraints that must be enforced inside the KNN query (not
post-filtered), e.g. language availability.

embeddingkit supports this via:

- `search.Options.FilterSQL`
- `search.Options.FilterArgs` (named args referenced as `@name`)

This fragment is appended as `AND (<FilterSQL>)`. It is trusted SQL owned by the
host app.

## Optional helpers

These are optional and should not be required for core usage:

- `search.MMRReRank(...)` diversity helper (caller supplies candidate-to-candidate similarity).
- `eval.RecallAtK(...)` and `eval.MRR(...)` metrics skeleton.

## Dead-letter queue (DLQ)

Non-retryable failures (or tasks that exceed max-attempts) are moved out of
`embedding_tasks` into:

- `embedding_dead_letters`

This keeps `embedding_tasks` mostly empty in steady state.

## Removing models (manual maintenance)

embeddingkit is config-driven. If a model is removed from the host app config:

- embeddingkit will stop enqueueing new tasks for it and stop using it for search
  (because the host app won't call it anymore),
- but embeddingkit will NOT automatically delete old embeddings or drop indexes.

If you want to clean up a removed model, you can do it manually:

- Delete stored vectors:

  - `DELETE FROM <schema>.embedding_vectors WHERE model = '<model>';`

- Drop per-model indexes created by `pg.EnsureModelIndexes`:

  - Find them by name pattern:
    - `idx_embedding_vectors_hnsw_cosine__*`
    - `idx_embedding_vectors_hnsw_binary__*`

  - Or query Postgres catalog to list matching indexes for your schema/table and
    drop them explicitly.
