# API surface — unified interface, ingestion & modes

This is the linchpin of the dual-mode design (see [DESIGN.md](DESIGN.md)): **one Go interface** that
the embedded core, the HTTP server, and the remote client SDK all implement. Host code programs
against the interface and is identical whether the hub is embedded or remote — only the constructor
changes.

> Sketch, not final signatures. Names/shapes will firm up during implementation; the point is the
> seam.

## The facade

```go
// Hub is the single surface host apps use, in either deployment mode.
type Hub interface {
    // --- Content plane (ingestion is PUSH; see below) ---
    UpsertEntity(ctx context.Context, e EntityDoc) error
    DeleteEntity(ctx context.Context, ref EntityRef) error

    // --- Content plane (query) ---
    Search(ctx context.Context, text string, opts SearchOptions) ([]Hit, error)
    Typeahead(ctx context.Context, text string, opts TypeaheadOptions) ([]Hit, error)
    SimilarTo(ctx context.Context, ref EntityRef, opts SimilarOptions) ([]Hit, error) // "more like this": vector + lexical + co-engagement

    // --- Signal plane ---
    RecordSignal(ctx context.Context, s Signal) error            // view-session / reaction / rating

    // --- Discovery plane (reads over entities × signals) ---
    History(ctx context.Context, subject Subject, opts HistoryOptions) ([]StateRow, error)
    Unseen(ctx context.Context, subject Subject, opts UnseenOptions) ([]EntityRef, error)
    States(ctx context.Context, subject Subject, refs []EntityRef) (map[EntityRef]State, error) // bulk UI annotation: seen %, resume
    Engagement(ctx context.Context, ref EntityRef) (EntityEngagement, error)
    Popular(ctx context.Context, entityType string, opts PopularOptions) ([]Hit, error)         // windowed rankings
    Recommend(ctx context.Context, subject Subject, opts RecOptions) ([]Hit, error)             // "for you" (user -> items)
}
```

Every method is implicitly **tenant-scoped**: in embedded mode the tenant is pinned at construction;
in server mode it is resolved from the request's authkit token (see Tenancy).

## Core value types

```go
type EntityRef struct { Tenant, EntityType, EntityID string }

type EntityDoc struct {
    EntityRef
    Language    string
    Lexical     string             // FTS / trigram text   (host-built, pushed)
    Semantic    string             // text to embed         (host-built, pushed)
    AssetURLs   []AssetURL         // VL image/frame URLs    (host-built, pushed)
    // Visibility — the only catalog primitives the hub interprets:
    VisibleFrom time.Time          // host maps its publish/live time
    Deleted     bool
    // Filtering — host-defined facets; the hub assigns NO meaning (mechanism vs meaning):
    Flags       []string           // set-membership: "premium", "r18", region tags, ...
    Attrs       map[string]string  // key/value facets
    NumAttrs    map[string]float64 // numeric facets for range filters
}

type Subject struct { Tenant string; UserID *string; AnonKey *string } // one of UserID / AnonKey

// State = one subject's standing with one entity (for UI annotation: progress bar + resume).
type State struct {
    Seen         bool
    MaxProgress  uint32    // progress bar = MaxProgress / ProgressMax
    ProgressMax  uint32
    Completed    bool
    Resume       string    // opaque host pointer: last page / scroll / timestamp
    LastSignalAt time.Time
}

type Signal struct {
    EntityRef
    Subject
    Type        string             // HOST-DEFINED: "view" | "rate" | "purchase" | "listen" | ...
    OccurredAt  time.Time
    DurationS   uint32             // active time (optional)
    Progress    uint32             // consumption numerator (pages / scroll% / watched-s) — optional
    ProgressMax uint32             // consumption denominator — optional
    Value       float64            // explicit feedback: rating / vote / price / score — optional
    Label       string             // categorical: reaction kind / variant — optional
    Weight      float64            // host/scorer-assigned importance (default 1)
    Payload     map[string]any     // anything else
}
```

## Extension interfaces (host-provided, per entity type)

```go
type Scorer interface { // see signal-plane.md
    Score(ctx context.Context, s Session) (score uint8, progress, progressMax uint32, completed bool)
}

type EntityCatalog interface {
    // The "universe" for Unseen: live, non-deleted entity ids of a type for a tenant.
    // Implemented as a synced dim table or a reader; defines visibility/filtering rules.
    Universe(ctx context.Context, tenant, entityType string, opts CatalogQuery) ([]string, error)
}

type Registry interface {
    Register(entityType string, scorer Scorer, catalog EntityCatalog, opts EntityTypeOptions)
}
```

## Ingestion is push, not pull-callbacks

Searchkit today pulls content via `BuildLexicalString` / `BuildSemanticDocument` / `ListAssetURLs`
callbacks — embedded-only. The unified hub flips this: hosts **push** content with `UpsertEntity`
(and signals with `RecordSignal`), so both deployment modes are symmetric. Internally the worker
still drives `search_dirty` → backfill → `embedding_tasks`; `UpsertEntity` writes the document and
marks it dirty.

```go
// On entity create/update, the host pushes (embedded: direct call; server: HTTP):
hub.UpsertEntity(ctx, EntityDoc{
    EntityRef: EntityRef{Tenant: "doujins", EntityType: "blog_post", EntityID: "1234"},
    Language:  "en",
    Lexical:   title + " " + body,
    Semantic:  title + "\n" + summary,
    Metadata:  map[string]any{"live_at": liveAt, "est_read_time_s": 240},
})
```

## Two constructors, one interface

```go
// Embedded: in-process, single implicit tenant.
hub, _ := hubkit.NewEmbedded(hubkit.Config{
    PG: pgPool, CH: chConn, Schema: "doujins",
    Tenant: "doujins",            // TenantsEnabled=false
    Embedder: embedder,
})

// Remote: identical interface over HTTP; tenant comes from the authkit token.
hub := hubkit.NewRemote("https://hub.internal", hubkit.RemoteConfig{
    TokenSource: authkitTokenSource, // carries tenant claim
})
```

`NewEmbedded` and `NewRemote` both return `Hub`. Swapping modes is a one-line wiring change.

## Server mode

A thin HTTP (optionally gRPC) server exposes each `Hub` method as an endpoint, authenticates via
authkit (token → tenant), and delegates to the same core. The remote client SDK marshals the same
types. No business logic lives in the transport layer — it is a pass-through over the core.

## Compatibility note

The existing `searchkit.Client` (`Search` / `Typeahead` / `SimilarTo` with pull-callbacks +
host-built `QueryVec`-free text API) is the seed of the embedded content-plane path. The unified
`Hub` supersedes it; the content-plane methods map onto today's `client.go`, and the signal/discovery
methods are new. No backwards-compat layer is a goal (consistent with prior searchkit cleanups).
