# Recommender eval (offline)

Python implementation of the recommendations-eval protocol: global temporal split, full-catalog ranking per warm user (seen-train excluded), recall@10/20 + NDCG@10, head/tail stratification. It exists so any candidate recommender is judged against tuned baselines on the same yardstick before it earns its way in (the Dacrema RecSys'19 lesson: tuned baselines beat most learned models).

The bench consumes a normalized interaction log (see the input contract below); adapters that produce it from a concrete data source are gitignored and live with their host.

## Layout

    protocol.py        the judge: load, temporal split, metrics, evaluate()
    baselines.py       the bar: popularity + co-visitation (--grid, --eyeball)
    diagnostics.py     audits: history starvation, scoring parity, ensembles,
                       browse-adjacency share of hits
    candidates/        challengers, one file per model
      item2vec.py      gensim skip-gram (--sessions for session corpus)
      als.py           implicit-ALS (--sessions for session x item matrix)
      gorse_similarity.py   IDF-shrunk-cosine co-visitation (gorse's formula)
    adapters/          GITIGNORED - host-side data converters
    data/              GITIGNORED - working csv/tsv files
    .venv/             GITIGNORED

## Input contract (RECOMMENDER_DATA dir, default ./data)

- `interactions.csv` — `user_id,item_id,timestamp,weight`. One row per positive interaction. The host decides what counts as positive and maps signal strength to `weight` (1.0 weak, >1.0 strong). Ids are opaque strings; timestamps must sort chronologically as strings.
- `sessions.csv` (optional) — `session_id,item_id,timestamp`. Ordered events incl. repeats; the session-granular training corpus. Consumers truncate to events before the temporal cut (leakage guard).
- `labels.csv` (optional) — `item_id,labels`. Free-text labels, only for the qualitative neighbor eyeball.

## Running

    python3 -m venv .venv && .venv/bin/pip install numpy scipy gensim implicit
    .venv/bin/python baselines.py            # the bar
    .venv/bin/python baselines.py --grid     # tune it hard
    .venv/bin/python candidates/item2vec.py --sessions
    .venv/bin/python candidates/als.py --sessions
    .venv/bin/python candidates/gorse_similarity.py
    .venv/bin/python diagnostics.py

## Recorded runs

### dataset A — engagement snapshot

848K favorites/downloads/views, 2023, user-keyed, no sessions; 6.2K eval users, 33K catalog.

| model | recall@10 | recall@20 | ndcg@10 |
|---|---|---|---|
| popularity | 0.067 | 0.122 | 0.053 |
| covis-raw cap=20 topk=100 (BAR) | 0.128 | 0.157 | 0.108 |
| ALS best (f=128 a=10 weighted) | 0.066 | 0.094 | 0.050 |
| item2vec best (win=20) | 0.030 | 0.046 | 0.024 |

### dataset B — view stream

20.9M views over 30 days, sessionized to 3.9M sessions; 90K eval users, 53K catalog, 94% anonymous.

| model | recall@20 | ndcg@10 | tail-hit@20 |
|---|---|---|---|
| popularity | 0.048 | 0.030 | 0.000 |
| covis-raw cap=20 topk=100 (BAR) | 0.088 | 0.043 | 0.001 |
| item2vec on sessions (best) | 0.073 | 0.036 | 0.002 |
| ALS on sessions (best) | 0.035 | 0.018 | 0.004 |
| gorse-sim (any lambda, +/-IDF) | 0.081 | 0.039 | — |
| covis cosine-normalized | 0.079 | 0.038 | — |
| RRF covis+item2vec | 0.083 | 0.042 | — |

### diagnostics (dataset B)

- 31% of eval users have <=3 train items (19% exactly 1) — a third of the eval is near-cold for every method.
- Scoring parity: item2vec with covis-style kNN-sum scoring got WORSE (0.064) than the centroid (0.073) — the loss is not a scoring artifact.
- Ensembles: every RRF blend scored below covis alone — the candidates add no complementary signal on this data.
- Browse adjacency: 26-28% of hits are within +/-10 item-ids of history for ALL methods equally (the "next upload in the feed" pattern).

### Conclusions (as of 2026-07-28)

Raw-count co-visitation with a ~20-item recency window and top-100 neighbor truncation beat 15 challengers across two datasets: learned models, ensembles, and normalized-similarity variants. On short-window, thin-history, anonymous data, popularity is signal — every technique that normalizes it away (cosine, IDF, shrinkage, latent factors) measurably loses. item2vec went from 29% of the bar (dataset A, wrong-shaped) to 83% (dataset B, sessions): the data-shape hypothesis is real, and the candidates' rematch conditions are long windows + stable identities + engagement-typed feedback — i.e. post-launch data. The tail (~majority of test volume) is unserved by ALL behavioral methods (hit-rate ~0.00x): tail/cold coverage needs a content channel, not a better behavioral model.

Caveats: gensim 4.4.0 on numpy 2.5 emitted BLAS-shim warnings (training completed, full coverage); ALS is scored by item-factor aggregation rather than fold-in, so its numbers are a floor.
