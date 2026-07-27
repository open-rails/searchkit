"""Candidate models on the eval protocol: item2vec (gensim, Caselles-Dupre tuning)
and implicit-ALS, judged on the SAME temporal split + metrics as the baselines
(eval.py). Ship rule: a candidate must beat the tuned co-vis baseline
overall AND on the tail, else co-vis stays the interim system.

Host-agnostic input: interactions.csv (user_id,item_id,timestamp,weight) —
see eval.py. Rows with weight>1 are treated as "strong" for the weighted-ALS
variant; the host decides what earns that weight.
"""
import os, csv, math, time, collections, sys
import numpy as np
from scipy import sparse

DATA = os.environ.get('RECOMMENDER_DATA', os.path.dirname(os.path.abspath(__file__)))
K_EVAL = (10, 20)
HIST_FOR_USER_VEC = 20    # matches the winning baseline's recency window

t0 = time.time()

first = {}
strong = set()
with open(f'{DATA}/interactions.csv') as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        if len(row) < 3:
            continue
        u, i, ts = row[0], row[1], row[2]
        w = float(row[3]) if len(row) > 3 and row[3] else 1.0
        key = (u, i)
        if key not in first or ts < first[key]:
            first[key] = ts
        if w > 1.0:
            strong.add(key)

events = sorted(((ts, u, i) for (u, i), ts in first.items()))
cut = events[int(len(events) * 0.8)][0]
train_by_user = collections.defaultdict(list)
test_by_user = collections.defaultdict(set)
for ts, u, i in events:
    (train_by_user[u].append(i) if ts < cut else test_by_user[u].add(i))

item_pop = collections.Counter(i for h in train_by_user.values() for i in h)
items = sorted(item_pop)
item_ix = {it: k for k, it in enumerate(items)}
n_items = len(items)
eval_users = []
for u, tset in test_by_user.items():
    h = train_by_user.get(u)
    if not h:
        continue
    truth = {i for i in tset if i in item_ix and i not in set(h)}
    if truth:
        eval_users.append((u, truth))
head_cutoff = sorted(item_pop.values(), reverse=True)[max(1, n_items // 10) - 1]
is_head = {i: item_pop[i] >= head_cutoff for i in items}
print(f'prep: catalog={n_items:,} eval_users={len(eval_users):,} ({time.time()-t0:.0f}s)')

def evaluate(name, user_score_fn):
    rec = {k: [] for k in K_EVAL}
    nd, hh, hn, th, tn = [], 0, 0, 0, 0
    for u, truth in eval_users:
        scores = user_score_fn(u)
        if scores is None:
            scores = np.zeros(n_items, dtype=np.float32)
        seen = [item_ix[i] for i in set(train_by_user[u]) if i in item_ix]
        scores[seen] = -np.inf
        top = np.argpartition(scores, -20)[-20:]
        top = top[np.argsort(scores[top])[::-1]]
        tix = {item_ix[i] for i in truth}
        for k in K_EVAL:
            rec[k].append(len(tix & set(top[:k].tolist())) / min(len(tix), k))
        dcg = sum(1 / math.log2(r + 2) for r, j in enumerate(top[:10]) if j in tix)
        idcg = sum(1 / math.log2(r + 2) for r in range(min(10, len(tix))))
        nd.append(dcg / idcg if idcg else 0)
        t20 = set(top[:20].tolist())
        for i in truth:
            if is_head[i]:
                hn += 1; hh += item_ix[i] in t20
            else:
                tn += 1; th += item_ix[i] in t20
    print(f'{name:34} recall@10={np.mean(rec[10]):.3f} recall@20={np.mean(rec[20]):.3f} '
          f'ndcg@10={np.mean(nd):.3f} | head-hit@20={hh/max(hn,1):.3f} tail-hit@20={th/max(tn,1):.3f} ({tn} tail)')

# ══ item2vec (gensim) ══
from gensim.models import Word2Vec

sentences = [[str(i) for i in h] for h in train_by_user.values() if len(h) >= 2]
print(f'item2vec corpus: {len(sentences):,} user sequences')

def item2vec_run(ns_exponent, window, epochs=30, dims=100):
    m = Word2Vec(sentences, vector_size=dims, window=window, min_count=1, sg=1,
                 negative=10, ns_exponent=ns_exponent, sample=1e-4,
                 workers=8, epochs=epochs, seed=7)
    V = np.zeros((n_items, dims), dtype=np.float32)
    hitn = 0
    for i in items:
        k = str(i)
        if k in m.wv:
            V[item_ix[i]] = m.wv[k]; hitn += 1
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    def fn(u):
        hist = [i for i in train_by_user[u][-HIST_FOR_USER_VEC:] if i in item_ix]
        if not hist:
            return None
        uv = Vn[[item_ix[i] for i in hist]].mean(axis=0)
        n = np.linalg.norm(uv)
        if n < 1e-9:
            return None
        return (Vn @ (uv / n)).astype(np.float32)
    return fn, hitn

for ns in (0.75, 0.0, -0.5):
    for window in (5, 20):
        fn, cov = item2vec_run(ns, window)
        evaluate(f'item2vec ns={ns:+.2f} win={window} (cov={cov})', fn)
        sys.stdout.flush()

# ══ implicit-ALS ══
from implicit.als import AlternatingLeastSquares

def als_run(factors, reg, alpha, weighted):
    rows, cols, vals = [], [], []
    uix = {u: k for k, u in enumerate(train_by_user)}
    for u, h in train_by_user.items():
        for i in h:
            w = 3.0 if (weighted and (u, i) in strong) else 1.0
            rows.append(uix[u]); cols.append(item_ix[i]); vals.append(w)
    UI = sparse.csr_matrix((np.array(vals, np.float32), (rows, cols)),
                           shape=(len(uix), n_items))
    model = AlternatingLeastSquares(factors=factors, regularization=reg,
                                    alpha=alpha, iterations=20, random_state=7)
    model.fit(UI, show_progress=False)
    IF = model.item_factors
    UF = model.user_factors
    def fn(u):
        k = uix.get(u)
        if k is None:
            return None
        return (IF @ UF[k]).astype(np.float32)
    return fn

for factors, alpha, weighted in ((64, 40, False), (128, 40, False), (128, 40, True), (128, 10, True)):
    fn = als_run(factors, 0.01, alpha, weighted)
    evaluate(f'ALS f={factors} a={alpha} w={weighted}', fn)
    sys.stdout.flush()

print(f'\ntotal {time.time()-t0:.0f}s')
