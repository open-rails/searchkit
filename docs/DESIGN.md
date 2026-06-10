# searchkit — system design

> Status: design / north-star. This document describes where searchkit is going: from a
> hybrid-search library into a **unified entity + signal platform** that powers **search,
> recommendations, history, "unseen", and engagement tracking** from one place — delivered **first as
> an embedded library** (the immediate scope); a standalone, multi-tenant **SaaS** service is the
> future direction.
>
> Companion docs: [signal-plane.md](signal-plane.md) (the new per-user layer),
> [api-surface.md](api-surface.md) (the unified interface). Implementation notes live in
> [../agents/NOTES.md](../agents/NOTES.md); the embedded work breakdown is in
> [../agents/progress.json](../agents/progress.json), and the standalone/SaaS plans in
> [../agents/future.json](../agents/future.json).

## 1. What searchkit is today

A Go library for Postgres-backed **hybrid search**, already keyed by `(entity_type, entity_id,
language)`:

- **Lexical** — `pg_trgm` typeahead + Postgres FTS (`tsvector`) + PGroonga for `ja/zh/ko`.
- **Semantic** — pgvector `halfvec` embeddings; plus **VL** (image/visual embeddings via asset URLs).
- **Fusion/diversity** — Reciprocal Rank Fusion (RRF) + MMR.
- **`SimilarTo(entity)`** — nearest-neighbour similarity (a content-based recommendation primitive).
- **Worker** — `worker.SyncOnce` consumes `search_dirty`, runs resumable backfill, drains
  `embedding_tasks`.
- Content ingested via **host pull-callbacks** (`BuildLexicalString`, `BuildSemanticDocument`,
  `ListAssetURLs`).

## 2. Where it's going (and why one system)

Search answers "which entities match this query." But the same product needs three more things, and
they all key off the *same* entities:

- **History** — what has this user already seen?
- **Unseen** — what new entities has this user *not* seen? (surface fresh content)
- **Engagement** — how good was a view / what did the user think? (quality, not raw counts)
- **Recommendations** — what should we show this user next?

The insight that makes this **one** system rather than four: introduce a **signal layer** —
`(subject, entity_type, entity_id) → signals` — and every feature becomes a read over *entities ×
signals*:

| Capability | Reads |
|---|---|
| Search (personalized) | content rank ⊕ signal affinity + popularity |
| Recommendations | `SimilarTo` (content) ⊕ collaborative filtering over the signal matrix |
| History | signal current-state by subject, ordered by recency |
| Unseen | entity catalog **minus** the subject's seen-set |
| Engagement | the signal stream + per-entity aggregates |
| Popularity / trending | windowed aggregates over the signal stream, ranked — fixed windows (30/90/365d) **and** arbitrary date slices |

So search and recommendations are the same matrix read from two directions; history and unseen are
membership queries against it. That's the whole reason to build it as a single library.

## 3. Architecture: three planes

1. **Content plane** (exists) — entity registry + lexical/semantic/visual indexes + fusion +
   `SimilarTo`.
2. **Signal plane** (new) — append-only event stream of view-sessions / reactions / ratings, plus a
   durable per-`(subject, entity)` **current-state** projection. See
   [signal-plane.md](signal-plane.md).
3. **Discovery plane** (the join) — personalized search, recommendations, history, unseen.

A raw "view count" is deliberately **not** a goal; it is a trivial aggregate over the signal stream
if anyone ever wants it.

**Governing principle — the hub owns *mechanism*, the host owns *meaning*.** Storage, indexing,
aggregation, and the four queries are the hub's. Entity types, signal types, facets, scoring weights,
and filters are *host-defined data* — never columns the hub understands. No business noun (`premium`,
`pages`, `reading`) appears in the schema; app-specific concepts live in host-defined facets, signal
types, and scorers. This is what lets any content service adopt it.

Each entity type a host registers (e.g. `gallery`, `blog_post`, `video`, `track`, `product`) provides:

- **Indexable content** — lexical/semantic text + VL asset URLs, **pushed** on change (see §6).
- **Scorer** — maps a raw signal payload → engagement score + progress + completion. This is where
  entity types differ (the hub stays generic):

  | App · entity | Scorer ("quality of an interaction") |
  |---|---|
  | doujins · gallery | pages viewed / page_count (completed ≥ 90%) |
  | doujins · blog post | read-time + scroll depth (completed ≥ 90% scroll or ≥ est read-time) |
  | hentai0 · video | watch-time / % watched |
  | (any) · track / product / lesson | play % / purchase / quiz score — host-defined |

- **Facets** — host-defined filterable attributes (`flags` / `attrs` / `num_attrs`) the hub stores
  but assigns no meaning to; used by *unseen*/search/recs filtering. Gating like "premium", "region",
  "age-rating" maps onto these — the hub never learns what they mean.
- **Visibility** — the only universal catalog primitives the hub interprets: `visible_from` +
  `deleted`.
- id type.

## 5. Deployment modes

One **transport-agnostic core** behind **one Go interface** (see [api-surface.md](api-surface.md));
only the constructor differs. **Embedded is the immediate scope; the standalone server is future**
(see [../agents/future.json](../agents/future.json)).

- **Embedded library (now)** — host builds the core with its own DB pools and calls it in-process
  (`NewEmbedded(...)`), as `searchkit.NewClient` does today. It runs against the **shared DB** (a
  dedicated schema), returns **ranked entity ids** (+ per-user `State` on enrich), and the host
  **hydrates ids → cards** (thumbnails, links) from its own DB. No network hop, and no presentation
  data duplicated into searchkit. This is the fast, simple path and covers doujins + hentai0's needs.
- **Standalone server (later)** — two topologies:
  - **A: frontend → api-server → searchkit** — the api-server calls searchkit over HTTP for ids, then
    hydrates locally. One server-to-server hop; for polyglot/multi-app or independent scaling.
  - **B: frontend → searchkit** (openrails-style) — searchkit answers the frontend **directly** with
    full cards. Removes the api-server from the read path, but requires **render payloads** (it must
    store thumbnail URL / link / title) + **hub-side facet gating** (no app server in the loop to
    filter) + token auth. Trades data duplication for a removed hop.

  Both reuse the same `Hub` via a remote client SDK (`NewRemote(url)`) — host code is identical.

**Why embedded first:** standalone only buys multi-app/polyglot, independent scaling, or a
frontend-direct read path — none of which doujins needs to *use* searchkit. Embedded + id-returning is
fastest and duplicates nothing.

## 6. Ingestion

- **Embedded (now):** keep searchkit's existing **pull-callbacks** (`BuildLexicalString` /
  `BuildSemanticDocument` / `ListAssetURLs`) for content, and a host-provided `EntityCatalog` that
  reads the host's own tables for the unseen universe + gating. Nothing is duplicated — the host's
  data stays in the host's DB. The signal plane is push by nature (`RecordSignal`).
- **Standalone (later):** a server can't call back into the host's Go funcs, so it switches to **push
  ingestion** — the host `UpsertEntity(...)`s lexical/semantic text + facets (+ optional render
  payload). See [../agents/future.json](../agents/future.json). The worker (`search_dirty` → backfill
  → `embedding_tasks`) is unchanged; only how it's fed differs.

## 7. Tenancy & SaaS

> Multi-tenant server mode is **future** ([../agents/future.json](../agents/future.json)); the
> embedded library is **single-tenant** now. The `tenant_id` column exists even embedded (one value)
> so the two share code.

Modeled on authkit (`TenantSlug` / `TenantID` / `TenantsEnabled`, tenant-scoped delegated tokens),
which the doujins/hentai0 stack already uses.

- **Server mode** — multi-tenant SaaS. Each request carries an authkit token whose claim identifies
  the tenant; the core scopes every read/write to it. doujins and hentai0 are **separate tenants**
  with isolated entities, signals, search, and recommendations.
- **Embedded mode** — single implicit tenant (`TenantsEnabled=false`); the host app is the sole
  tenant.

Consequences:

- **Tenant is a first-class key**: `(tenant, entity_type, entity_id)` across entities, documents,
  embeddings, signals, current-state, recs.
- **Per-tenant isolation by default** — one tenant's behaviour never influences another's
  recs/search ranking. Cross-tenant is a possible opt-in later.
- **Auth** reuses authkit delegated tokens.

**Open decision — isolation mechanism** (spans Postgres + ClickHouse):

- *Schema-per-tenant* (Postgres) matches the open-rails precedent (`SET LOCAL search_path`) and gives
  hard isolation, but ClickHouse has no equivalent and it scales poorly to many tenants.
- *Uniform `tenant_id` column* (both stores) is consistent and scales; ClickHouse favours a tenant
  column anyway (leading sort/partition key) — at the cost of relying on the core to inject the
  tenant filter on every query.

Lean: uniform `tenant_id` column with centralized enforcement in the core; schema-per-tenant
Postgres as a stronger-isolation option for a few large tenants.

## 8. Storage

- **Placement** — the hub owns a **dedicated Postgres schema** (configurable; *not* the host app's
  schema, to avoid colliding with app tables like `entities` / `search_documents`) and a **dedicated
  ClickHouse database**. Embedded = that one schema/db is the single implicit tenant; server = one
  schema/db with many tenants by `tenant_id` column (not schema-per-tenant). Host `FilterSQL` still
  references app tables by qualified name (same database, cross-schema).
- **Content plane** — Postgres (`pg_trgm` / `tsv` / pgvector / PGroonga), as today.
- **Signal plane** — event stream → ClickHouse (high-volume append; one event per session/interaction,
  **never per read**); a daily per-entity rollup (`entity_daily`) for **popularity/trending** over any
  window + per-user affinity; durable current-state (no TTL). See [signal-plane.md](signal-plane.md)
  for the store split, the unseen anti-join, and the popularity rollup.
- **Identity** — authkit (subject = resolved user id; anonymous = session-key hash). The hub does not
  own identity.

The hub spans two stores; the goal is to unify the **API**, not the storage.

## 9. Phased rollout

The embedded work breakdown lives in [../agents/progress.json](../agents/progress.json); the
standalone/SaaS plans in [../agents/future.json](../agents/future.json). Sequencing rationale:
history/unseen/engagement are nearly free off the signal plane and immediately useful; personalized
search and collaborative recommendations are the heavy lifts and come **after** the substrate is
proven across ≥2 consumers.

1. **Signal plane** — event stream + current-state + `Scorer` interface.
2. **History / unseen / engagement** read APIs.
3. **Push ingestion** for the content plane (server-mode ready).
4. **Multi-tenancy** (tenant_id everywhere; `TenantsEnabled`).
5. **Deployment modes** — unified interface, embedded core, HTTP server, remote client SDK.
6. **Personalized search** — signal-aware ranking.
7. **Recommendations** — content-based + collaborative.
8. **Consumer rollout** — doujins blog posts (pilot; delete the legacy blog view counter) →
   doujins galleries (migrate existing analytics onto the hub) → hentai0 videos.

## 10. Naming

The repo is `searchkit` today. As it grows past search into the unified hub, it may be renamed (it is
self-contained and not owned by open-rails). Treat "searchkit" in these docs as the evolving system,
not just its search plane.
