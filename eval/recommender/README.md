# Recommender eval (offline)

Python implementation of the recommendations-eval protocol:
global temporal split, full-catalog ranking per warm user (seen-train
excluded), recall@10/20 + NDCG@10, head/tail stratification. It exists so any
candidate recommender (item2vec/ALS, or anything later) is judged
against tuned baselines on the same yardstick before it earns its way in
(the Dacrema RecSys'19 lesson: tuned baselines beat most learned models).

searchkit stays agnostic: the bench consumes a normalized interaction log and
attaches no meaning to ids, signal types, or labels. Everything host-specific
lives in a HOST-side adapter that produces the inputs (the adapter used for
the recorded run below lives with its host, not in this repo).

## Input contract (RECOMMENDER_DATA dir, default = script dir; all gitignored)

- `interactions.csv` — `user_id,item_id,timestamp,weight`
  One row per positive interaction. The HOST decides what counts as positive
  and maps signal strength to `weight` (1.0 weak, >1.0 strong). Ids are opaque
  strings; timestamps must sort chronologically as strings.
- `labels.csv` (optional) — `item_id,labels`
  Free-text labels used only for the qualitative neighbor eyeball.

## Scripts

- `eval.py` — the protocol + baselines (popularity, co-visitation raw/cosine).
- `tune.py` — the "tuned hard" grid (history cap x neighbor topK x
  all-vs-strong positives) + neighbor eyeball.
- `models.py` — the candidate models: item2vec (gensim, Caselles-Dupre
  ns_exponent sweep) and implicit-ALS (factors/alpha/weighted), judged on the
  identical protocol.

## Recorded run (2026-07-27, one host's 848K-interaction snapshot, 4 weeks, 2023)

| model | recall@10 | recall@20 | ndcg@10 | tail-hit@20 |
|---|---|---|---|---|
| popularity | 0.067 | 0.122 | 0.053 | 0.000 |
| covis-raw cap=20 topk=100 (BAR) | 0.128 | 0.157 | 0.108 | ~0 |
| ALS best (f=128 a=10 weighted) | 0.066 | 0.094 | 0.050 | 0.000 |
| item2vec best (win=20) | 0.030 | 0.046 | 0.024 | 0.003 |

Both learned candidates LOSE to the tuned co-vis baseline. Per the ship
rule, co-visitation is the interim system; embeddings iterate when real
SESSION-granular data exists (this snapshot is user-keyed/deduped over a
short window — no sessions, which is precisely what item2vec needed).
Tuning lessons: recency cap ~20 items beats longer histories; RAW co-counts
beat cosine normalization on a short window; strong signals work as weights,
not filters; nobody cracks the tail.

## Running

1. Produce `interactions.csv` (+ optional `labels.csv`) with your host-side
   adapter, drop them next to the scripts (gitignored) or point `RECOMMENDER_DATA`
   at their dir.
2. `python3 -m venv .venv && .venv/bin/pip install numpy scipy gensim implicit`
3. Run in order: `eval.py` (baselines), `tune.py` (grid), `models.py`
   (candidates). Full suite is ~5 minutes on CPU.

Caveat: gensim 4.4.0 on numpy 2.5 emitted BLAS-shim warnings
("our_dot_float") — training completed with full coverage, but item2vec
numbers may not be its absolute best.
