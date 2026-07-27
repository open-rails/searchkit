# Recs eval prototype (searchkit #24 protocol, offline)

Python implementation of the #24 recommendations-eval protocol, run 2026-07-27
against the Gorse feedback snapshot (the owner's prior recommender: 848K
deduped interactions, 2023-08-28..09-25, doujins galleries). THE DATA IS NOT IN
THIS REPO (real user behavior — privacy) and neither is the venv; scripts only.

## Scripts

- `recs_eval.py` — the protocol: global temporal split (80/20), full-catalog
  ranking per warm user (seen-train excluded), recall@10/20 + NDCG@10,
  head/tail stratification. Baselines: popularity, item co-visitation
  (raw + cosine).
- `recs_tune.py` — the "tuned hard" grid (history cap x neighbor topk x
  positives-set) + qualitative neighbor eyeball using the gorse items' tag
  labels.
- `recs_models.py` — the two #36 candidates: item2vec (gensim, Caselles-Dupre
  ns_exponent sweep) and implicit-ALS (factors/alpha/strong-signal-weighting),
  judged on the identical protocol.

## Recorded verdict (2026-07-27)

| model | recall@10 | recall@20 | ndcg@10 | tail-hit@20 |
|---|---|---|---|---|
| popularity | 0.067 | 0.122 | 0.053 | 0.000 |
| covis-raw cap=20 topk=100 (BAR) | 0.128 | 0.157 | 0.108 | ~0 |
| ALS best (f=128 a=10 weighted) | 0.066 | 0.094 | 0.050 | 0.000 |
| item2vec best (win=20) | 0.030 | 0.046 | 0.024 | 0.003 |

Both learned candidates LOSE to the tuned co-vis baseline (the Dacrema
RecSys'19 lesson, reproduced on our own data). Per #36's ship rule the co-vis
approach is the interim system; embeddings iterate when real SESSION data
exists (the gorse snapshot is user-keyed/deduped with a 4-week window — no
sessions, which is precisely what item2vec needed). Tuning lessons: recency
cap ~20 items beats longer histories; RAW co-counts beat cosine normalization
on a short window; strong signals (favorite/download) work as weights, not
filters. Nobody cracks the tail (~0 hit-rate on 32K tail test items).

## Reproducing

1. Data: export from the gorse MySQL (phpMyAdmin scripted export works; see
   doujins tracker #790/#791 era notes):
   `SELECT feedback_type,user_id,item_id,time_stamp FROM gorse.feedback` (CSV),
   plus `items`/`users`, plus a legacy `folders(id,main,parent_id,is_sub,
   is_trashed)` dump as `folders_index.tsv` (tab-separated) for the
   gallery-vs-series classification (galleries = `main=1 AND parent_id>1`;
   ids are preserved by legacy-migrate, so `folder-N` == gallery id N).
2. Drop `feedback_all.csv`, `items.csv`, `folders_index.tsv` NEXT TO the
   scripts (this dir; they are gitignored) — or point `RECS_DATA` at any dir
   holding them.
3. `python3 -m venv .venv && .venv/bin/pip install numpy scipy gensim implicit`
   (`.venv` here is gitignored too).
4. Run in order: `recs_eval.py` (baselines), `recs_tune.py` (grid),
   `recs_models.py` (candidates). Full suite is ~5 minutes on CPU.

Caveat: gensim 4.4.0 on numpy 2.5 emitted BLAS-shim warnings
("our_dot_float") — training completed with full coverage, but item2vec
numbers may not be its absolute best.
