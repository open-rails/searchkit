"""Tuning grid for the co-vis baseline ('baselines to beat, tuned
hard') + qualitative neighbor eyeball.

Host-agnostic inputs (RECOMMENDER_DATA dir, default = this dir):
  interactions.csv  user_id,item_id,timestamp,weight  (see eval.py)
  labels.csv        item_id,labels  (optional; opaque host labels, any text —
                    used only to eyeball whether neighbor lists look coherent)

Grid: per-user history cap x neighbor topK x positives set (ALL rows vs
weight>1 "strong" rows only).
"""
import os, csv, math, time, collections, itertools
import numpy as np
from scipy import sparse

DATA = os.environ.get('RECOMMENDER_DATA', os.path.dirname(os.path.abspath(__file__)))
K_EVAL = (10, 20)

raw = []   # (user, item, ts, weight)
with open(f'{DATA}/interactions.csv') as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        if len(row) < 3:
            continue
        w = float(row[3]) if len(row) > 3 and row[3] else 1.0
        raw.append((row[0], row[1], row[2], w))

def build(strong_only):
    first = {}
    for u, i, ts, w in raw:
        if strong_only and w <= 1.0:
            continue
        key = (u, i)
        if key not in first or ts < first[key]:
            first[key] = ts
    ev = sorted(((ts, u, i) for (u, i), ts in first.items()))
    cut = ev[int(len(ev) * 0.8)][0]
    tr_u = collections.defaultdict(list); te_u = collections.defaultdict(set)
    for ts, u, i in ev:
        (tr_u[u].append(i) if ts < cut else te_u[u].add(i))
    pop = collections.Counter(i for u, h in tr_u.items() for i in h)
    items = sorted(pop); ix = {it: k for k, it in enumerate(items)}
    users = []
    for u, ts_ in te_u.items():
        h = tr_u.get(u)
        if not h:
            continue
        truth = {i for i in ts_ if i in ix and i not in set(h)}
        if truth:
            users.append((u, truth))
    return tr_u, users, pop, items, ix

def run(tr_u, users, pop, items, ix, cap, topk, tag):
    n = len(items)
    rows, cols = [], []
    uix = {u: k for k, u in enumerate(tr_u)}
    for u, h in tr_u.items():
        for i in h[-cap:]:
            rows.append(uix[u]); cols.append(ix[i])
    A = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(uix), n))
    A.data[:] = 1
    C = (A.T @ A).tocsr()
    pv = np.asarray(A.sum(0)).ravel()
    D = sparse.diags(1 / np.sqrt(np.maximum(pv, 1)))
    COS = (D @ C @ D).tocsr()
    def spar(S):
        S = S.tolil(); S.setdiag(0); S = S.tocsr()
        for r in range(S.shape[0]):
            lo, hi = S.indptr[r], S.indptr[r+1]
            if hi - lo > topk:
                idx = np.argpartition(S.data[lo:hi], -topk)[-topk:]
                keep = np.zeros(hi-lo, bool); keep[idx] = True
                S.data[lo:hi][~keep] = 0
        S.eliminate_zeros(); return S
    for vname, S in {'raw': spar(C), 'cos': spar(COS)}.items():
        rec10, rec20, nd = [], [], []
        for u, truth in users:
            h = tr_u[u]
            hix = [ix[i] for i in h[-cap:] if i in ix]
            sc = np.asarray(S[hix].sum(0)).ravel().astype(np.float32)
            seen = [ix[i] for i in set(h) if i in ix]
            sc[seen] = -np.inf
            top = np.argpartition(sc, -20)[-20:]
            top = top[np.argsort(sc[top])[::-1]]
            tix = {ix[i] for i in truth}
            rec10.append(len(tix & set(top[:10].tolist())) / min(len(tix), 10))
            rec20.append(len(tix & set(top[:20].tolist())) / min(len(tix), 20))
            dcg = sum(1/math.log2(r+2) for r, j in enumerate(top[:10]) if j in tix)
            idcg = sum(1/math.log2(r+2) for r in range(min(10, len(tix))))
            nd.append(dcg/idcg if idcg else 0)
        print(f'  {tag} cap={cap:<4} topk={topk:<4} {vname:4} '
              f'recall@10={np.mean(rec10):.3f} recall@20={np.mean(rec20):.3f} ndcg@10={np.mean(nd):.3f}')

print('=== GRID: all positives ===')
tr_u, users, pop, items, ix = build(strong_only=False)
print(f'(warm users={len(users):,}, catalog={len(items):,})')
for cap, topk in itertools.product((20, 50, 200), (50, 100, 500)):
    run(tr_u, users, pop, items, ix, cap, topk, 'ALL')

print('=== STRONG-only (weight>1 rows) ===')
tr_s, users_s, pop_s, items_s, ix_s = build(strong_only=True)
print(f'(warm users={len(users_s):,}, catalog={len(items_s):,})')
for cap, topk in itertools.product((50, 200), (100, 500)):
    run(tr_s, users_s, pop_s, items_s, ix_s, cap, topk, 'STR')

# ── qualitative: neighbor lists with host labels (optional) ──
labels_path = f'{DATA}/labels.csv'
if os.path.exists(labels_path):
    print('\n=== NEIGHBOR EYEBALL (covis-cos, all positives, cap=50 topk=100) ===')
    labels = {}
    with open(labels_path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) >= 2:
                labels[row[0]] = row[1]
    n = len(items)
    rows, cols = [], []
    uix = {u: k for k, u in enumerate(tr_u)}
    for u, h in tr_u.items():
        for i in h[-50:]:
            rows.append(uix[u]); cols.append(ix[i])
    A = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(uix), n))
    A.data[:] = 1
    C = (A.T @ A).tocsr()
    pv = np.asarray(A.sum(0)).ravel()
    D = sparse.diags(1/np.sqrt(np.maximum(pv, 1)))
    COS = (D @ C @ D).tolil(); COS.setdiag(0); COS = COS.tocsr()
    popular = [items[j] for j in np.argsort(pv)[::-1][:200]]
    import random
    random.seed(7)
    for src in random.sample(popular, 4):
        j = ix[src]
        row = COS[j].toarray().ravel()
        nb = np.argsort(row)[::-1][:4]
        print(f'\n  SRC {src}  [{labels.get(src, "")[:90]}]')
        for b in nb:
            if row[b] <= 0:
                break
            bid = items[b]
            print(f'    -> {bid} sim={row[b]:.2f} [{labels.get(bid, "")[:80]}]')
else:
    print('\n(no labels.csv — skipping neighbor eyeball)')
