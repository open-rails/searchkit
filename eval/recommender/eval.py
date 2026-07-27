"""Recommender eval protocol, offline reference implementation.

Host-agnostic: consumes a normalized interaction log produced by a HOST-side
adapter — searchkit attaches no meaning to user/item ids or to what the host
counted as a positive signal.

Input (in RECOMMENDER_DATA dir, default = this dir):
  interactions.csv  header: user_id,item_id,timestamp,weight
    - one row per positive interaction event; the host decides what counts as
      positive and maps its signal strengths to `weight` (>=1.0; e.g. 1.0
      weak, 3.0 strong). Timestamps must sort chronologically as strings
      (ISO-8601 or zero-padded epoch, one style throughout).

Protocol: global TEMPORAL split (train = first 80% of time), full-catalog
ranking per warm user (seen-train excluded), recall@10/20 + NDCG@10,
head/tail stratification. Baselines: popularity, item co-visitation
(raw + cosine) — the bar any learned model must beat (Dacrema RecSys'19).
"""
import os, csv, math, time, collections
import numpy as np
from scipy import sparse

DATA = os.environ.get('RECOMMENDER_DATA', os.path.dirname(os.path.abspath(__file__)))
TRAIN_FRACTION = 0.8
HISTORY_CAP = 50          # per-user items used for co-vis matrix (most recent)
TOPK_NEIGHBORS = 100      # sparsify item-item sims
K_EVAL = (10, 20)

t0 = time.time()

# load + dedupe to first occurrence per (user, item)
first = {}
with open(f'{DATA}/interactions.csv') as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        if len(row) < 3:
            continue
        u, i, ts = row[0], row[1], row[2]
        key = (u, i)
        if key not in first or ts < first[key]:
            first[key] = ts
events = sorted(((ts, u, i) for (u, i), ts in first.items()))
print(f'interactions (user,item deduped): {len(events):,}  '
      f'[{events[0][0]} .. {events[-1][0]}]  ({time.time()-t0:.0f}s)')

cut = events[int(len(events) * TRAIN_FRACTION)][0]
train = [(u, i) for ts, u, i in events if ts < cut]
test = [(u, i) for ts, u, i in events if ts >= cut]
print(f'split at {cut}: train={len(train):,} test={len(test):,}')

train_by_user = collections.defaultdict(list)   # ordered (time) by construction
for u, i in train:
    train_by_user[u].append(i)
test_by_user = collections.defaultdict(set)
for u, i in test:
    test_by_user[u].add(i)

# item universe = items with >=1 train interaction (rankable catalog)
item_pop = collections.Counter(i for _, i in train)
items = sorted(item_pop)
item_ix = {it: k for k, it in enumerate(items)}
n_items = len(items)

# warm eval users: >=1 train item and >=1 UNSEEN test item in catalog
eval_users = []
for u, tset in test_by_user.items():
    hist = train_by_user.get(u)
    if not hist:
        continue
    truth = {i for i in tset if i in item_ix and i not in set(hist)}
    if truth:
        eval_users.append((u, truth))
cold_users = sum(1 for u in test_by_user if u not in train_by_user)
print(f'catalog={n_items:,} items | eval users={len(eval_users):,} | cold (excluded)={cold_users:,}')

# head/tail: head = top 10% items by train popularity
head_cutoff = sorted(item_pop.values(), reverse=True)[max(1, n_items // 10) - 1]
is_head = {i: (item_pop[i] >= head_cutoff) for i in items}

# ── build sparse train matrix (capped history for co-vis) ──
rows, cols = [], []
user_ids = list(train_by_user)
user_ix = {u: k for k, u in enumerate(user_ids)}
for u, hist in train_by_user.items():
    for i in hist[-HISTORY_CAP:]:
        rows.append(user_ix[u]); cols.append(item_ix[i])
A = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(len(user_ids), n_items))
A.data[:] = 1.0  # binarize duplicates
print(f'train matrix {A.shape}, nnz={A.nnz:,}  ({time.time()-t0:.0f}s)')

def topk_sparsify(S, k):
    S = S.tolil()
    S.setdiag(0)
    S = S.tocsr()
    for r in range(S.shape[0]):
        lo, hi = S.indptr[r], S.indptr[r + 1]
        if hi - lo > k:
            idx = np.argpartition(S.data[lo:hi], -(k))[-k:]
            keep = np.zeros(hi - lo, dtype=bool); keep[idx] = True
            S.data[lo:hi][~keep] = 0
    S.eliminate_zeros()
    return S

C = (A.T @ A).tocsr()                     # raw co-occurrence
pop_vec = np.asarray(A.sum(axis=0)).ravel()
inv_sqrt = 1.0 / np.sqrt(np.maximum(pop_vec, 1))
D = sparse.diags(inv_sqrt)
COS = (D @ C @ D).tocsr()                 # cosine (popularity-damped)
C = topk_sparsify(C, TOPK_NEIGHBORS)
COS = topk_sparsify(COS, TOPK_NEIGHBORS)
print(f'sims built: raw nnz={C.nnz:,} cos nnz={COS.nnz:,}  ({time.time()-t0:.0f}s)')

pop_scores = np.array([item_pop[i] for i in items], dtype=np.float32)

def evaluate(name, score_fn):
    recall = {k: [] for k in K_EVAL}
    ndcg10, head_hits, head_n, tail_hits, tail_n = [], 0, 0, 0, 0
    for u, truth in eval_users:
        hist = train_by_user[u]
        scores = score_fn(u, hist)
        seen = [item_ix[i] for i in set(hist) if i in item_ix]
        scores[seen] = -np.inf
        kmax = max(K_EVAL)
        top = np.argpartition(scores, -kmax)[-kmax:]
        top = top[np.argsort(scores[top])[::-1]]
        truth_ix = {item_ix[i] for i in truth}
        for k in K_EVAL:
            hits = len(truth_ix & set(top[:k].tolist()))
            recall[k].append(hits / min(len(truth_ix), k))
        dcg = sum(1 / math.log2(r + 2) for r, ix in enumerate(top[:10]) if ix in truth_ix)
        idcg = sum(1 / math.log2(r + 2) for r in range(min(10, len(truth_ix))))
        ndcg10.append(dcg / idcg if idcg else 0)
        top20 = set(top[:20].tolist())
        for i in truth:
            if is_head[i]:
                head_n += 1; head_hits += item_ix[i] in top20
            else:
                tail_n += 1; tail_hits += item_ix[i] in top20
    print(f'{name:12} recall@10={np.mean(recall[10]):.3f} recall@20={np.mean(recall[20]):.3f} '
          f'ndcg@10={np.mean(ndcg10):.3f} | head-hit@20={head_hits/max(head_n,1):.3f} ({head_n}) '
          f'tail-hit@20={tail_hits/max(tail_n,1):.3f} ({tail_n})')

def pop_fn(u, hist):
    return pop_scores.copy()

def make_covis_fn(S):
    def fn(u, hist):
        ix = [item_ix[i] for i in hist[-HISTORY_CAP:] if i in item_ix]
        if not ix:
            return np.zeros(n_items, dtype=np.float32)
        return np.asarray(S[ix].sum(axis=0)).ravel().astype(np.float32)
    return fn

print(f'\n=== BASELINES (temporal split, full catalog, {len(eval_users):,} warm users) ===')
evaluate('pop', pop_fn)
evaluate('covis-raw', make_covis_fn(C))
evaluate('covis-cos', make_covis_fn(COS))
print(f'\ntotal {time.time()-t0:.0f}s')
