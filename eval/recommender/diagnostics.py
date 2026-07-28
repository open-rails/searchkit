"""Audits that stress the verdict, not the models:

- history starvation: how thin are eval users' train histories?
- scoring parity: item2vec judged with co-vis's own machinery (kNN-truncated
  neighbor sum) vs the centroid dot product
- ensembles: RRF blends — if a candidate adds complementary signal, the blend
  beats the best ingredient; if not, it dilutes it
- browse adjacency: share of hits within +/-10 item-ids of history (the
  "next upload in the feed" artifact), reported for every method

  python diagnostics.py
"""
import sys

import numpy as np
from scipy import sparse

from baselines import covis_matrix, covis_fn, DEFAULT_CAP, DEFAULT_TOPK
from protocol import Bench, load_sessions

EKNN_CHUNK = 2048


def history_starvation(bench):
    hl = sorted(len(bench.train_by_user[u]) for u, _ in bench.eval_users)

    def pct(a, p):
        return a[int(p * (len(a) - 1))]

    print(f'history length of eval users: median={pct(hl, .5)} '
          f'p25={pct(hl, .25)} p75={pct(hl, .75)} p90={pct(hl, .9)} | '
          f'==1: {100 * sum(1 for x in hl if x == 1) / len(hl):.0f}% '
          f'<=3: {100 * sum(1 for x in hl if x <= 3) / len(hl):.0f}%')
    tl = sorted(len(t) for _, t in bench.eval_users)
    print(f'truth size: median={pct(tl, .5)} p90={pct(tl, .9)}')
    sys.stdout.flush()


def train_item2vec(bench):
    from gensim.models import Word2Vec
    sessions = load_sessions(bench)
    sentences = [s for s in sessions.values() if len(s) >= 2]
    m = Word2Vec(sentences, vector_size=100, window=5, min_count=3, sg=1,
                 negative=10, ns_exponent=0.0, sample=1e-4, workers=14,
                 epochs=10, seed=7)
    V = np.zeros((bench.n_items, 100), dtype=np.float32)
    for i in bench.items:
        if i in m.wv:
            V[bench.item_ix[i]] = m.wv[i]
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)


def embedding_knn(Vn, topk=100):
    n = Vn.shape[0]
    rows, cols, vals = [], [], []
    for s in range(0, n, EKNN_CHUNK):
        e = min(s + EKNN_CHUNK, n)
        sims = Vn[s:e] @ Vn.T
        for rloc in range(e - s):
            row = sims[rloc]
            row[s + rloc] = -1
            idx = np.argpartition(row, -topk)[-topk:]
            good = idx[row[idx] > 0]
            rows.extend([s + rloc] * len(good))
            cols.extend(good.tolist())
            vals.extend(row[good].tolist())
    return sparse.csr_matrix(
        (np.array(vals, np.float32), (rows, cols)), shape=(n, n))


def centroid_fn(bench, Vn, cap=DEFAULT_CAP):
    def fn(u):
        hix = [bench.item_ix[i] for i in bench.train_by_user[u][-cap:]
               if i in bench.item_ix]
        if not hix:
            return None
        uv = Vn[hix].mean(0)
        n = np.linalg.norm(uv)
        return None if n < 1e-9 else (Vn @ (uv / n)).astype(np.float32)
    return fn


def sparse_sum_fn(bench, S, cap=DEFAULT_CAP):
    def fn(u):
        hix = [bench.item_ix[i] for i in bench.train_by_user[u][-cap:]
               if i in bench.item_ix]
        if not hix:
            return None
        return np.asarray(S[hix].sum(0)).ravel().astype(np.float32)
    return fn


def rrf(fns, n_items, k=60, depth=200):
    def fn(u):
        out = np.zeros(n_items, np.float32)
        got = False
        for f in fns:
            sc = f(u)
            if sc is None:
                continue
            got = True
            order = np.argsort(sc)[::-1][:depth]
            for r, j in enumerate(order):
                out[j] += 1.0 / (k + r + 1)
        return out if got else None
    return fn


def main():
    bench = Bench()
    bench.describe()
    history_starvation(bench)

    C = covis_matrix(bench, DEFAULT_CAP, DEFAULT_TOPK)
    cov = covis_fn(bench, C, DEFAULT_CAP)
    Vn = train_item2vec(bench)
    E = embedding_knn(Vn)
    cen = centroid_fn(bench, Vn)
    eknn = sparse_sum_fn(bench, E)

    bench.evaluate('covis (bar)', cov, adjacency=True)
    bench.evaluate('item2vec centroid', cen, adjacency=True)
    bench.evaluate('item2vec kNN100', eknn, adjacency=True)
    bench.evaluate('RRF covis+centroid', rrf([cov, cen], bench.n_items))
    bench.evaluate('RRF covis+eknn', rrf([cov, eknn], bench.n_items))
    bench.evaluate('RRF covis+eknn+pop',
                   rrf([cov, eknn, lambda u: bench.pop_scores.copy()],
                       bench.n_items))


if __name__ == '__main__':
    main()
