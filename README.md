# searchkit

`searchkit` is a Go library for:

- **Typeahead / fuzzy lexical search** (language-specific) via Postgres `pg_trgm` over `search_documents.document` (and optionally PGroonga for `ja/zh/ko`).
- **Keyword lexical search** (BM25-family; language-specific) via Postgres full-text search over `search_documents.tsv` (and optionally PGroonga for `ja/zh/ko`).
- **Semantic search** (language-specific embeddings) via pgvector `halfvec` stored in `embedding_vectors`.
- A **single, host-run worker loop** that:
  - consumes `search_dirty` notifications (changed/deleted entities),
  - runs resumable cursor-based backfill (no “insert 10M dirty rows”),
  - and drains `embedding_tasks` to compute/store embeddings.

This README is a **manual** for host applications. Design notes live in `agents/NOTES.md`.

> **Status & direction.** searchkit is evolving from a hybrid-search library into a **unified entity
> + signal platform** — search + recommendations + history + "unseen" + engagement — runnable both
> **embedded** and as a **multi-tenant SaaS server**. Canonical design:
> [`docs/DESIGN.md`](docs/DESIGN.md), [`docs/signal-plane.md`](docs/signal-plane.md),
> [`docs/api-surface.md`](docs/api-surface.md). The **embedded hub** (signal + discovery planes) is
> implemented — see "The embedded hub" below. The standalone SaaS server is future work.

## The embedded hub (signal + discovery planes)

Beyond search, searchkit can run as an in-process **discovery hub**: it records per-user
interaction **signals** in ClickHouse and answers id-returning discovery queries — history, unseen,
view-context annotation, engagement, popularity/trending, personalized search, and
recommendations. The host hydrates ids → cards from its own DB; searchkit stores no presentation
data.

**Mechanism vs meaning:** searchkit owns storage/aggregation/queries. Entity types, signal types,
scoring weights, and completion rules are host-defined data — no business noun appears in the
schema.

### Setup

The hub needs the existing Postgres content plane plus a **dedicated ClickHouse database**:

```go
import (
	"github.com/ClickHouse/clickhouse-go/v2"
	"github.com/open-rails/searchkit"
	"github.com/open-rails/searchkit/signal"
)

ch, _ := clickhouse.Open(&clickhouse.Options{Addr: []string{"localhost:9000"}})

// Once at startup (DDL privileges required). Idempotent. Set Cluster for
// replicated/ON CLUSTER deployments.
_ = signal.EnsureSchema(ctx, ch, signal.SchemaOptions{Database: "hub"})

hub, _ := searchkit.NewEmbedded(searchkit.EmbeddedConfig{
	PG:           pgPool,
	PGSchema:     "hub",          // dedicated schema (NOT the host app schema)
	Embedder:     embedder,
	DefaultModel: "qwen3-embedding",
	CH:           ch,
	CHDatabase:   "hub",          // dedicated ClickHouse database
	Tenant:       "myapp",        // single implicit tenant embedded
	Scorers: map[string]signal.Scorer{
		// Host-defined: map a raw session to score/progress/completed.
		"blog_post": blogScorer, // e.g. read-time + scroll depth, completed >= 90%
	},
	Catalogs: map[string]searchkit.EntityCatalog{
		// Host-defined: the "universe" for Unseen, read from YOUR tables
		// with YOUR visibility/premium gating, newest first.
		"blog_post": blogCatalog,
	},
})
```

Omitting `CH` runs content-only: search/typeahead work, signal/discovery methods return
`ErrSignalPlaneDisabled`.

### Recording signals

One summarized event per session/interaction (exit-beacon style — never one row per scroll tick):

```go
_ = hub.RecordSignal(ctx, signal.Signal{
	EntityRef:   signal.EntityRef{EntityType: "blog_post", EntityID: "42"},
	Subject:     signal.Subject{UserID: userID},   // or AnonKey for anonymous
	Type:        "view",                            // host-defined
	DurationS:   180,
	Progress:    95, ProgressMax: 100,              // generic consumption units
	Resume:      "scroll:95",                       // opaque resume pointer
})
```

The registered `Scorer` for the entity type fills `Score`/`Progress`/`ProgressMax`/`Completed`.
Replayed events (same content or same `EventID`) deduplicate instead of double-counting.

### Discovery reads (all id-returning)

```go
hist, _ := hub.History(ctx, user, signal.HistoryOptions{EntityType: "blog_post", Status: signal.HistoryInProgress})
fresh, _ := hub.Unseen(ctx, user, searchkit.UnseenOptions{EntityType: "blog_post", Limit: 20})
states, _ := hub.States(ctx, user, refs)          // bulk "seen? % read? resume?" for a page of cards
eng, _   := hub.Engagement(ctx, ref)              // unique subjects, completion rate, avg score
top, _   := hub.Popular(ctx, "blog_post", signal.PopularOptions{Window: signal.LastDays(30)})
slice, _ := hub.Popular(ctx, "blog_post", signal.PopularOptions{Window: signal.Between(a, b)}) // arbitrary date slices
recs, _  := hub.Recommend(ctx, user, searchkit.RecommendOptions{EntityTypes: []string{"blog_post"}})
```

- `Popular` merges a tiny daily rollup (`entity_daily`) for day-aligned windows and scans raw
  events for sub-day slices. Default ranking: log-scaled unique subjects × Bayesian-smoothed
  engagement; tune via `RankWeights` (incl. `HalfLifeDays` time decay) or replace with a trusted
  `RankExpr`.
- `Recommend` fuses content similarity (seeded from the subject's high-signal entities) with
  co-engagement, excludes seen, and falls back to popularity on cold start. **Negative feedback
  demotes**: signals with `Value < 0` (e.g. a dislike) exclude an entity from seeds and results,
  count *against* co-engagement strength, and (in personalized search) apply a strong
  `DislikePenalty`. Co-engagement reads the precomputed `item_pairs` rollup when present — refresh
  it periodically via `hub.RefreshCoEngagement(...)` — and falls back to a query-time scan.
  Optional MMR diversity via `DiversityLambda` (uses stored embeddings; best-effort).
- `hub.Search(..., HubSearchOptions{Personalize: &searchkit.Personalization{Subject: user, DemoteSeen: true}})`
  blends candidate popularity into the ranking and demotes seen/completed entities — recall
  unchanged, ranking only, per-request toggle.
- `hub.SimilarTo(..., HubSimilarOptions{CoEngagement: true})` fuses vector neighbours with
  "subjects who engaged with X also engaged with Y".

## Host app integration (manual)

### 1) Apply Postgres migrations (required)

searchkit migrations are applied/tracked with migratekit (`public.migrations`) under `app=searchkit`,
and are scoped to the host schema via `SET LOCAL search_path = <schema>, public`.

Note on PGroonga (CJK/Korean support):

- You must install the PGroonga extension package in your Postgres image for your Postgres major version (package names vary by distro).
  - Example (Debian/Ubuntu images): install `postgresql-<MAJOR>-pgroonga` from the PGDG/APT repo, then restart Postgres.
- The baseline migration runs `CREATE EXTENSION pgroonga`, which typically requires superuser (or elevated) privileges.
- If your environment can’t run `CREATE EXTENSION` from app migrations, install/enable PGroonga out-of-band, then mark the migration applied (or apply it manually).
- If PGroonga is not installed/enabled, CJK/Korean routing (`ja/zh/ko`) will fail at query time with a Postgres error (missing operator/function/index).

```go
import (
	"context"
	"database/sql"

	"github.com/doujins-org/migratekit"
	"github.com/open-rails/searchkit/migrations"
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

> ⚠️ **Changing (new design).** Pull-callbacks (`BuildLexicalString` / `BuildSemanticDocument` /
> `ListAssetURLs`) are being replaced by **push ingestion**: the host will call `UpsertEntity(...)`
> with the lexical/semantic text + asset URLs. Pull-callbacks only work *embedded*; push makes the
> embedded and server modes symmetric. See [`docs/api-surface.md`](docs/api-surface.md). The callback
> model below is how it works **today**.

Host apps provide:

- `runtime.BuildSemanticDocument(ctx, entity_type, language, []entity_id) -> map[id]string` (**required**)
  - Used to generate embeddings.
- `runtime.BuildLexicalString(ctx, entity_type, language, []entity_id) -> map[id]string` (required if you want lexical docs)
  - Used to populate `search_documents` for both trigram typeahead and FTS.
- `vl.ListAssetURLs(ctx, entity_type, []entity_id) -> map[id][]AssetURL` (required only if VL models are enabled)

### 4) Mark changes (host writes `search_dirty`)

> ⚠️ **Changing (new design).** With push ingestion the host calls `UpsertEntity` and searchkit marks
> `search_dirty` internally — hosts will no longer write `search_dirty` directly. Current behavior
> below.

The host does **not** enqueue per-model tasks directly.
Instead, it upserts into `<schema>.search_dirty`:

- `(entity_type, entity_id, language, is_deleted, reason, updated_at)`

searchkit decides what to rebuild based on worker config + active model set.

### 5) Run one worker loop (host-owned, searchkit-provided)

Run a background worker (River/cron/goroutine) that calls:

- `worker.SyncOnce(ctx, rt, worker.SearchkitOptions{...})`

This single entrypoint:

1) processes `search_dirty`,
2) runs bounded backfill for missing docs/embeddings,
3) drains `embedding_tasks` (does provider calls and writes `embedding_vectors`).

### 6) Query candidates (lexical + semantic)

Recommended entrypoint:

- Create a SearchKit client once and reuse it:

```go
client, err := searchkit.NewClient(searchkit.ClientConfig{
  Pool:            pool,
  Schema:          "doujins",
  Embedder:        rt,          // runtime.Runtime implements Embedder
  DefaultModel:    "text-embed-3-small",
  DefaultLanguage: "en",
})
```

Then per request:

```go
hits, err := client.Search(ctx, userQuery, searchkit.SearchOptions{
  Language: "en",
  LanguageMode: searchkit.LanguageModeExact, // exact|fallback_en (default exact)
  Mode:     searchkit.SearchModeDual, // lexical|semantic|dual
  EntityTypes: []string{"gallery"},
  Limit:          20,  // final fused results
  CandidateLimit: 100, // per-source candidates before RRF; defaults to Limit
  // Positive cosine floor applied to semantic candidates before RRF.
  // Values <= 0 disable the additional floor.
  SemanticMinSimilarity: 0.35,
})
```

Search scores and semantic confidence are different domains:

- `SearchHit.Score` is an RRF score derived from source ranks. It is not cosine similarity.
- `SemanticMinSimilarity` filters raw cosine similarity before RRF.
- `SemanticMinSimilarityEnabled` makes an explicit zero floor inclusive, retaining zero-similarity candidates while dropping negative candidates.
- `SimilarOptions.MinSimilarityEnabled` provides the same explicit zero-floor behavior for nearest-neighbor recommendations.
- `CandidateLimit` controls each source's retrieval depth; `Limit` controls the final response size.
- `OversampleFactor` controls only the binary-quantized first stage when `TwoStage=true`. Values `<=1` use the effective default `5`.

For offline evaluation and diagnostics, use the opt-in traced call:

```go
hits, trace, err := client.SearchWithTrace(ctx, userQuery, opts)
```

The trace records effective routing/configuration, source candidates and score domains, and exact RRF contributions. On failure, it contains the work completed before the error. Ordinary `Search` does not collect trace candidates.

Traces contain normalized query text and candidate entity IDs. Treat them as potentially sensitive evaluation artifacts: do not log them indiscriminately, attach them to user-visible errors, or expose them from public APIs.

Typeahead suggestions while typing:

```go
hits, err := client.Typeahead(ctx, userQuery, searchkit.TypeaheadOptions{
  Language: "en",
  LanguageMode: searchkit.LanguageModeExact, // exact|fallback_en (default exact)
  EntityTypes: []string{"tag", "artist", "series"},
  Limit:    10,
  MinSimilarity: 0.3,
  FilterSQL:  "EXISTS (SELECT 1 FROM app.entities e WHERE e.id::text = sd.entity_id AND e.deleted_at IS NULL)",
  FilterArgs: map[string]any{},
})
```

Host-injected filters:

- `FilterSQL` and `FilterArgs` are supported on both `SearchOptions` and `TypeaheadOptions`.
- SearchKit applies these filters inside retrieval queries (before ranking/pagination) for lexical and semantic search paths.
- Treat `FilterSQL` as trusted host SQL only. Never concatenate raw user input into it; pass values through `FilterArgs`.
- This keeps SearchKit schema-agnostic: each host can enforce visibility/business constraints with host-specific SQL (including joins/EXISTS).
- In dual mode, one `FilterSQL` fragment is applied to both lexical (`sd`) and semantic (`ev`) queries. The fragment must therefore be valid in both SQL scopes. If policy needs backend-qualified aliases, issue explicit lexical/semantic calls with the matching fragment until channel-specific filter options are available.

Language strictness:

- `LanguageModeExact` (default): query only requested language.
- `LanguageModeFallbackEnglish`: query requested language and English in one call.
- Language mode is applied inside SearchKit retrieval (before ranking/pagination), not as post-filtering in host app code.

Language-specific routing (handled inside the client):

- For most languages, Typeahead uses `pg_trgm` over `<schema>.search_documents.document`, and Search uses Postgres FTS (`tsvector` + `ts_rank_cd`) for the lexical side.
- For `ja`/`zh`/`ko`, Typeahead and the lexical side of Search use **PGroonga** over `<schema>.search_documents.raw_document` (native-script), because Postgres FTS `simple` config does not provide Japanese/Chinese segmentation and trigram transliteration is lossy.
- Mixed-script (`CJK + ASCII`) queries run both lexical backends (PGroonga + trigram) and merge deterministically.

Query syntax notes:

- SearchKit does **not** treat leading `-term` as an operator. Leading `-` is treated as punctuation (so `-factor` behaves like `factor`).
- For Postgres FTS (`websearch_to_tsquery`), SearchKit normalizes intra-token hyphens to spaces so tokens like `two-factor` behave like `two factor`.
- Natural-language negation: for FTS only, `not X` is rewritten to `-X` before it reaches Postgres. This is a convenience for users typing normal phrases like `X not Y`.

## Offline search evaluation

Package `searchkit/eval` provides host-neutral golden-query evaluation:

- graded judgments (`0..3`) and exact-empty cases;
- recall@K, success@K, MRR@K, nDCG@K, and result-count metrics;
- versioned reports with dataset, suite, and candidate identities;
- caller-configured baseline tolerances;
- exact score-floor candidate generation and sweeps isolated by score domain.

Hosts still own corpus snapshots, business-policy fixtures, query execution, and production acceptance thresholds. Execution failures must be represented with `eval.Failed`; they are never treated as successful empty results.

```go
suite, err := eval.ParseSuite(fixture)
if err != nil {
  return err
}

outcome, err := eval.Evaluate(suite.Cases[0], []eval.Result{
  {Key: eval.GoldenKey{EntityType: "gallery", EntityID: "42"}, Score: 0.81},
})
if err != nil {
  return err
}

report, err := eval.BuildReport(eval.ReportIdentity{
  DatasetID: "corpus-snapshot-sha256",
  SuiteID: "suite-sha256",
  CandidateID: "searchkit-and-config-sha256",
}, []eval.Outcome{outcome}, "suite")
```

**Running a suite against a live client.** `eval` is dependency-free; wire your client behind `eval.CaseRunner` and let `eval.RunSuite` execute every case and build one report. `searchkit.NewEvalRunner` adapts a `*searchkit.Client` (each case's query → `client.Search` → results):

```go
runner := searchkit.NewEvalRunner(client, searchkit.SearchOptions{
  Mode: searchkit.SearchModeDual, Language: "en", SemanticMinSimilarity: 0.5,
})
report, err := eval.RunSuite(ctx, suite, runner, eval.ReportIdentity{
  DatasetID: "corpus-sha", SuiteID: suite.ID, CandidateID: "dual+floor0.5",
}, "query_type") // optional group-by labels → per-label breakdowns
```

**Regression gate.** Commit a baseline report as a golden file, then fail CI when quality drops beyond tolerance:

```go
comparison, err := eval.Compare(baseline, report, eval.Tolerances{
  RecallAtKDrop: 0, SuccessAtKDrop: 0, NDCGAtKDrop: 0.05, ExactEmptyRateDrop: 0,
})
if comparison.Regressed() { /* fail the test */ }
```

**Config diffing.** Run the suite under two `SearchOptions` (e.g. floor on vs off) with different `CandidateID`s and `Compare` the reports to pick a setting from measured nDCG/recall rather than by feel; `eval.CandidateFloors` + `eval.SweepResultFloors` sweep a semantic-score floor per score domain.

Quality metrics exclude execution failures from their denominators, while `FailedCases` remains explicit. `QualityStatus` distinguishes hits, misses, exact emptiness, unexpected results, and unjudged cases. Pass only stable lowercase identifier categories such as `timeout` or `semantic_search` to `eval.Failed`; unsafe categories normalize to `unspecified`. Reports include a deterministic content identity covering report identity, outcomes, ordered results, and scores. Baseline comparison validates that identity, report aggregates, and exact case definitions before comparing metrics. Floor sweeps require every outcome, including failures, to carry one matching `score_domain` label and accept `context.Context` for cancellation.

Before releasing changes to search retrieval or evaluation, run the PostgreSQL integration gate in addition to normal tests. Point it at a disposable test database: the test creates extensions and an isolated uniquely named schema, then removes that schema.

```bash
SEARCHKIT_TEST_URL='postgres://...' go test -count=1 -run TestClientSearch_Integration_LexicalAndSemantic .
```

Host integration details (contract, filter-builder patterns, hentai0/doujins examples):

- See `HOST_INTEGRATION.md`.

## Language → Postgres FTS config mapping

FTS uses a schema-local function created by migrations:

- `<schema>.searchkit_regconfig_for_language(language)`

It maps common codes like `en/es/fr/de/...` to built-in configs and falls back to `simple`.

## Model registry + ANN indexes

Construct the runtime via `runtime.NewWithContext(...)` to:

- upsert the configured model set into `<schema>.embedding_models`, and
- ensure per-model cosine + binary HNSW indexes exist (via `CREATE INDEX CONCURRENTLY`).
