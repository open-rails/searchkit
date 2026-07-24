# Signal plane — engagement, history & unseen

The signal plane is the genuinely new part of the system (see [DESIGN.md](DESIGN.md) for the whole
picture). It records **what each subject did and thought about each entity**, and projects that into
fast reads for history, unseen, engagement, and (later) personalized search + recommendations.

## Core idea

```
(tenant, subject, entity_type, entity_id) → signals
```

- **subject** — a resolved user id (from authkit) or, for anonymous traffic, a session key hash
  (`hash(ip|user_agent)` or a cookie id). History/unseen are meaningful only for logged-in subjects;
  anonymous signals still feed popularity/quality aggregates.
- **signals** — both *implicit* (view-session: duration, progress, completion) and *explicit*
  (reactions, ratings — "what they thought").

## Two tables

### 1. Event stream (append-only)

One row per view-session or interaction. **Never one row per read/scroll** — the host reports a
single summarized event when a session ends (on unload / visibility-hidden), exactly like the
existing gallery exit-beacon.

Conceptual columns:

```
tenant, entity_type, entity_id, subject,
signal_type        -- HOST-DEFINED: view | rate | purchase | listen | ...
occurred_at,
duration_s,        -- active time (optional)
progress,          -- consumption numerator (pages / scroll % / watch %) — optional
progress_max,      -- consumption denominator (page_count / 100 / video length) — optional
value,             -- explicit feedback: rating / vote / price / score — optional
label,             -- categorical: reaction kind / variant — optional
weight,            -- host/scorer-assigned importance (default 1)
score,             -- engagement 0..100 from the entity's Scorer
completed,         -- bool, per the entity's completion rule
payload            -- anything else (Map), for re-scoring later
```

The columns are **generic** (`progress`/`progress_max`), not gallery-specific (`max_page_reached`).
Each entity type's `Scorer` interprets them.

### 2. Current-state projection (durable, no TTL)

One row per `(tenant, subject, entity_type, entity_id)`, upserted from the event stream — the
load-bearing table for history/unseen/progress:

```
tenant, entity_type, entity_id, subject,
first_seen_at, last_signal_at,
total_events,
max_progress, progress_max, completed,   -- max_progress / progress_max = the "% watched" bar
resume,                                   -- opaque host pointer: last page / scroll / timestamp
has_interacted, last_score,
last_updated
```

On ClickHouse this is a `ReplacingMergeTree(last_updated)` ordered by
`(tenant, subject, entity_type, entity_id)` so the latest state wins; no TTL (durable seen-state,
unlike a 30-day visitor counter).

The projection is recomputed from the (deduplicated) event stream on every signal, so it is
replay-idempotent and self-heals on the next signal for a key. A crashed reprojection can leave a key
behind until then; a periodic host-scheduled sweep (`Hub.ReprojectStaleStates`) finds rows that lag the
stream and re-derives them.

## The Scorer (per-entity extension point)

```go
type Scorer interface {
    // Map a raw session payload to a normalized engagement score, generic progress,
    // and whether this counts as "completed".
    Score(ctx context.Context, s Session) (score uint8, progress, progressMax uint32, completed bool)
}
```

Examples:

- **gallery** — `progress = max_page_reached`, `progressMax = page_count`,
  `completed = progress ≥ ceil(0.9 * progressMax)`; score from pages + dwell.
- **blog post** — `progress = max_scroll_pct`, `progressMax = 100`,
  `completed = scroll ≥ 90 OR read_time ≥ est_read_time`; score from read-time + scroll.
- **video** — `progress = watched_seconds`, `progressMax = duration_s`,
  `completed = watched ≥ 0.9 * duration`; score from watch %.

The kit owns the storage/upsert/read machinery; the host owns only the `Scorer`.

## Impressions & attribution (learned-ranking data)

Beyond the engagement stream, a separate append-only table records **what was shown**, so clicks become
training labels for learned ranking:

- **`search_impressions`** — one row per SERP/shelf render (never per item). `tenant`, `query_id` (unique
  per render), `surface` (`search`/`foryou`/`similar`/`popular`/`organic`), `normalized_query` (normalized
  text only — no raw referrers/PII), `language`, `subject`, the shown items as parallel arrays
  `shown_entity_types` / `shown_entity_ids` / `shown_positions`, and `occurred_at`.
  `ReplacingMergeTree(recorded_at)`, month-partitioned; re-delivering a `query_id` deduplicates.

Write via `RecordImpressions` (batched, one row per render; shown entities are given in rank order and
positions are derived from `StartPosition`). A **click** is an ordinary signal that links
back to its render through standardized attribution payload keys — `query_id`, `surface`, `position` — set
via `Signal.WithAttribution(...)`. Training joins clicks to `search_impressions` on `query_id` and reads
the shown position, yielding `(query, shown items + positions, clicked item, dwell)`.

## Facets & filtering (host-defined)

`unseen` / search / recs accept a host-supplied filter over **generic facets** the hub stores but
assigns no meaning to — set-membership `flags` (e.g. `premium`, `r18`, region), key/value `attrs`,
numeric `num_attrs` — plus the two universal visibility primitives `visible_from` and `deleted`.
Gating like "premium" or "members-only" maps onto facets; the hub never learns what they mean
(mechanism vs meaning). Facets are mirrored alongside the catalog so the unseen anti-join can apply
them without calling back to the host.

## The four reads

### History — "what I've seen"
`current_state` for the subject, ordered by `last_signal_at DESC`. Filter by `completed` /
`in_progress` (via `max_progress` vs `progress_max`) for "finished" vs "started".

### Unseen — "new stuff I haven't seen"
`entity catalog (for type; visible_from <= now, not deleted, matching the host's facet filter) MINUS
{ entity_id : current_state row exists for subject with max_progress > 0 }`. Ordered by
`visible_from DESC`. This is the anti-join described below.

### Engagement — "quality of this entity"
Aggregates over the event stream per `(tenant, entity_type, entity_id)`: unique subjects, completion
rate, avg score, signal counts by type. Interaction-weighted quality, not raw view volume.

### Popularity & trending — "what's hot in a window"
Ranked entities over a time window, from a daily rollup `entity_daily` (AggregatingMergeTree, one row
per `(tenant, entity_type, entity_id, day)`: unique subjects, signals, engagement sum, completions,
per-type counts) fed by a materialized view from `signal_events`.

- **Fixed windows** (30/90/365d) = merge the last N day-buckets.
- **Arbitrary slices** (e.g. `2025-06-05 → 2025-08-08`) = merge the day-buckets in range.
- **Sub-day / custom** = scan `signal_events` directly (partitioned by month, so the range prunes).

`Popular(entity_type, window, filter, limit)` ranks by a default formula (log-scaled volume × avg
engagement, optional time-decay + Bayesian prior); the host can tune the weights or supply its own
ranking expression (mechanism vs meaning). The daily rollup is tiny (one row/entity/day), so it is
retained for years to serve "past year" and historical slices even if raw events expire sooner. The
same rollup feeds recs cold-start and search ranking.

### Annotations & resume — mark up lists with seen-state
For any list already on screen (search results, popular, similar, history), one bulk call
`States(subject, [entity_ids])` returns each entity's standing for that subject: seen?, `% =
max_progress / progress_max` (the YouTube-style progress bar), completed?, and `resume` (where to pick
up). It's a point/`IN` lookup on `current_state` (keyed by `(tenant, subject, entity_type,
entity_id)`), so annotating a page of cards is cheap. This is how seen-state is *displayed*; `Unseen`
is how it's *filtered out*.

### Similar & recommendations (two shapes)
Both read the same matrix; they differ by anchor:

- **Similar — "more like this" (item → items):** for a given entity, fuse content similarity
  (semantic `SimilarTo` + lexical/shared-tags) with **co-engagement** ("subjects who engaged with X
  also engaged with Y") from the signal matrix. Needs no logged-in user (works on a gallery page for
  anyone); optionally personalized by blending the viewer's affinity. RRF-fuse + MMR for diversity;
  drop the anchor entity (and optionally already-seen).
- **For you — (user → items):** collaborative filtering over the matrix + content from the subject's
  high-signal history; exclude seen; cold-start → popularity. See progress.json.

Personalized search is the same affinity/popularity signals fused into search ranking — a per-request
toggle the host flips (e.g. only for logged-in users), optionally demoting already-seen via
`current_state`.

## The unseen anti-join (store-split decision)

Unseen needs the **entity catalog** (content plane, Postgres) and the **seen-set** (signal plane,
ClickHouse) together. Options:

- **(a) Current-state in Postgres**, beside the registry → a clean single-store anti-join. Simplest
  at moderate volume; event stream can still live in ClickHouse.
- **(b) Catalog mirrored into ClickHouse** (the current doujins gallery approach) → a pure-ClickHouse
  anti-join, keeping high-volume reads in one analytical store.

Recommendation: events + aggregates in ClickHouse; resolve the anti-join via a synced catalog
(option b) unless a tenant's volume is low enough that option (a) is simpler. Decide per the tenancy
isolation choice in [DESIGN.md §7](DESIGN.md).

## What this replaces

In doujins, this generalizes the existing gallery analytics (`gallery_view_events`,
`user_history_current`, `gallery_interactions`) and **retires the blog "unique visitor" counter**
(`blog_post_visitors` + UV rollups). The counter was never the goal; durable per-user seen-state +
engagement is.
