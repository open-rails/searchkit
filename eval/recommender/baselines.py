"""The bar: non-learned baselines any candidate must beat (Dacrema RecSys'19 —
tuned baselines beat most learned models, so tune them hard first).

  python baselines.py            # popularity + co-visitation at the tuned config
  python baselines.py --grid     # cap x topk x positives-set tuning grid
  python baselines.py --eyeball  # qualitative neighbor lists (needs labels.csv)
"""
import argparse
import os
import random
import sys

import numpy as np
from scipy import sparse

from protocol import Bench, topk_sparsify

DEFAULT_CAP = 20
DEFAULT_TOPK = 100


def covis_matrix(bench, cap, topk, cosine=False):
    rows, cols = [], []
    uix = {u: k for k, u in enumerate(bench.train_by_user)}
    for u, h in bench.train_by_user.items():
        for i in h[-cap:]:
            rows.append(uix[u])
            cols.append(bench.item_ix[i])
    A = sparse.csr_matrix(
        (np.ones(len(rows), np.float32), (rows, cols)),
        shape=(len(uix), bench.n_items))
    A.data[:] = 1.0
    C = (A.T @ A).tocsr()
    if cosine:
        pv = np.asarray(A.sum(axis=0)).ravel()
        D = sparse.diags(1.0 / np.sqrt(np.maximum(pv, 1)))
        C = (D @ C @ D).tocsr()
    C = C.tolil()
    C.setdiag(0)
    C = C.tocsr()
    return topk_sparsify(C, topk)


def covis_fn(bench, S, cap):
    def fn(u):
        hix = [bench.item_ix[i] for i in bench.train_by_user[u][-cap:]
               if i in bench.item_ix]
        if not hix:
            return None
        return np.asarray(S[hix].sum(0)).ravel().astype(np.float32)
    return fn


def eyeball(bench, S, n_sources=4, n_neighbors=4, seed=7):
    labels_path = os.path.join(bench.data_dir, 'labels.csv')
    if not os.path.exists(labels_path):
        print('(no labels.csv — skipping neighbor eyeball)')
        return
    import csv
    labels = {}
    with open(labels_path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) >= 2:
                labels[row[0]] = row[1]
    popular = sorted(bench.item_pop, key=bench.item_pop.get, reverse=True)[:200]
    random.seed(seed)
    for src in random.sample(popular, n_sources):
        j = bench.item_ix[src]
        row = S[j].toarray().ravel()
        nb = np.argsort(row)[::-1][:n_neighbors]
        print(f'\n  SRC {src}  [{labels.get(src, "")[:90]}]')
        for b in nb:
            if row[b] <= 0:
                break
            bid = bench.items[b]
            print(f'    -> {bid} sim={row[b]:.2f} [{labels.get(bid, "")[:80]}]')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', action='store_true')
    ap.add_argument('--eyeball', action='store_true')
    ap.add_argument('--adjacency', action='store_true',
                    help='report browse-adjacency share of hits')
    args = ap.parse_args()

    bench = Bench()
    bench.describe()

    bench.evaluate('pop', lambda u: bench.pop_scores.copy())
    S = covis_matrix(bench, DEFAULT_CAP, DEFAULT_TOPK)
    bench.evaluate(f'covis-raw cap={DEFAULT_CAP} topk={DEFAULT_TOPK} (BAR)',
                   covis_fn(bench, S, DEFAULT_CAP), adjacency=args.adjacency)

    if args.grid:
        import itertools
        for strong_only in (False, True):
            b = Bench(strong_only=strong_only) if strong_only else bench
            tag = 'STRONG' if strong_only else 'ALL'
            if strong_only:
                b.describe()
            for cap, topk in itertools.product((20, 50, 200), (50, 100, 500)):
                for cosine in (False, True):
                    M = covis_matrix(b, cap, topk, cosine=cosine)
                    kind = 'cos' if cosine else 'raw'
                    b.evaluate(f'{tag} covis-{kind} cap={cap} topk={topk}',
                               covis_fn(b, M, cap))

    if args.eyeball:
        Scos = covis_matrix(bench, 50, DEFAULT_TOPK, cosine=True)
        eyeball(bench, Scos)


if __name__ == '__main__':
    sys.exit(main())
