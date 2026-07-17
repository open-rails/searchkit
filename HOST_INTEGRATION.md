# SearchKit Host Integration Guide

This guide defines the host-facing contract for embedding SearchKit in apps such as:

- `hentai0` (video search)
- `doujins` (gallery search)

SearchKit is a library. Hosts own HTTP routes, auth, and response shaping.

## Contract

Hosts should only use:

- `client.Search(ctx, query, searchkit.SearchOptions{...})`
- `client.SearchWithTrace(ctx, query, searchkit.SearchOptions{...})` for offline evaluation/debugging
- `client.Typeahead(ctx, query, searchkit.TypeaheadOptions{...})`

Use `SearchOptions.Mode`:

- `searchkit.SearchModeLexical`
- `searchkit.SearchModeSemantic`
- `searchkit.SearchModeDual`

Do not call low-level `searchkit/search` package APIs from host app code for normal request paths.

## Filter Policy (Host-Owned)

Hosts inject trusted SQL via:

- `FilterSQL string`
- `FilterArgs map[string]any`

Rules:

- Never concatenate raw user input into `FilterSQL`.
- Use named args in `FilterArgs`.
- Apply auth/business policy in host code and pass the resulting filter into SearchKit.
- For `SearchModeDual`, the same fragment runs in lexical (`sd`) and semantic (`ev`) SQL scopes. Use only a backend-neutral fragment. If a filter must reference those aliases, run explicit lexical and semantic calls with their respective fragments.

SearchKit applies filters in retrieval queries before ranking/pagination.

## Candidate Depth and Semantic Confidence

`SearchOptions` separates three controls:

- `Limit`: maximum final RRF-fused hits returned to the host.
- `CandidateLimit`: maximum candidates requested from each lexical/semantic source before RRF. It defaults to `Limit` and is clamped to at least `Limit`.
- `SemanticMinSimilarity`: positive raw cosine-similarity floor applied to semantic candidates before RRF. Values `<= 0` disable the additional floor; NaN and infinities are rejected.

`OversampleFactor` is independent: with two-stage semantic retrieval it controls the approximate stage-one width (`CandidateLimit * OversampleFactor`) before exact cosine rescore. Values `<= 1` use effective factor `5`.

Do not interpret `SearchHit.Score` as semantic confidence. It is an RRF rank score. Use `SearchWithTrace` to inspect raw source scores and their explicit score kinds. Trace collection is opt-in and returns partial provenance alongside errors.

Search traces include normalized query text and candidate IDs. Store them only in access-controlled evaluation artifacts; do not emit them wholesale to routine logs or user responses.

An explicitly configured `CandidateLimit` is capped at 10000. Existing `Limit`, `RRFK`, and `OversampleFactor` behavior remains compatible; two-stage multiplication and RRF arithmetic are checked or computed without integer overflow.

## Evaluation Ownership

SearchKit's `eval` package owns generic cases, metrics, reports, comparisons, and score-floor sweeps. Hosts own:

- immutable corpus/dataset identity;
- business-specific judgments and visibility policy;
- executing SearchKit and end-to-end host pipelines;
- cache isolation and failure policy;
- selecting production thresholds from measured sweeps.

Keep score domains separate when sweeping. In particular, do not combine cosine similarity, FTS rank, trigram similarity, PGroonga score, and RRF score.

## Language Strictness

Hosts pass request language explicitly (`Language`) and choose `LanguageMode`:

- `searchkit.LanguageModeExact` (default): requested language only.
- `searchkit.LanguageModeFallbackEnglish`: requested language + English.

For strict language behavior, set `LanguageModeExact` and keep host hydration/read-model queries strict as well (no implicit English fallback).

## Example: hentai0 (video search)

```go
const videoFilterSQL = `
EXISTS (
  SELECT 1
  FROM hentai0.videos v
  JOIN hentai0.video_versions vv ON vv.video_id = v.id
  WHERE v.id::text = sd.entity_id
    AND v.deleted_at IS NULL
    AND vv.deleted_at IS NULL
    AND (vv.live_at IS NULL OR vv.live_at <= NOW())
    AND v.default_version_id::uuid = vv.id::uuid
)`

hits, err := searchkitClient.Search(ctx, query, searchkit.SearchOptions{
  Language:     language,
  LanguageMode: searchkit.LanguageModeExact,
  Mode:         searchkit.SearchModeLexical,
  EntityTypes:  []string{"video"},
  Limit:        100,
  FilterSQL:    videoFilterSQL,
})
```

Host then hydrates `EntityID` values into API response cards.

## Example: doujins (gallery typeahead)

```go
filterSQL := `
EXISTS (
  SELECT 1
  FROM doujins.galleries g
  JOIN doujins.gallery_i18n gi
    ON gi.gallery_id = g.id
   AND gi.language = @language
  LEFT JOIN doujins.gallery_i18n_versions giv
    ON giv.id = gi.default_version_id
  WHERE g.id::text = sd.entity_id
    AND (@show_soft_deleted OR g.deleted_at IS NULL)
    AND (@show_soft_deleted OR gi.deleted_at IS NULL)
    AND (@show_soft_deleted OR giv.deleted_at IS NULL OR giv.id IS NULL)
    AND (@show_drafts OR giv.live_at IS NOT NULL OR giv.id IS NULL)
    AND (@show_future OR giv.live_at <= NOW() OR giv.id IS NULL)
)`

hits, err := searchkitClient.Typeahead(ctx, query, searchkit.TypeaheadOptions{
  Language:     language,
  LanguageMode: searchkit.LanguageModeExact,
  EntityTypes:  []string{"gallery"},
  Limit:        12,
  FilterSQL:    filterSQL,
  FilterArgs: map[string]any{
    "language":          language,
    "show_soft_deleted": permissionOptions.ShowSoftDeleted,
    "show_drafts":       permissionOptions.ShowDrafts,
    "show_future":       permissionOptions.ShowFuture,
  },
})
```

Host then resolves IDs to gallery payloads and applies response formatting.

## Migration Checklist

- Create and reuse a single `searchkit.Client`.
- Remove app-local lexical backend branching (FTS/PGroonga selection).
- Move app business filtering to host filter-builders (`FilterSQL/FilterArgs`).
- Keep app code to request validation, policy mapping, and hydration.
