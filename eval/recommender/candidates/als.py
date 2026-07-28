"""Candidate: implicit-ALS.

  python candidates/als.py             # user x item matrix (weighted variants)
  python candidates/als.py --sessions  # session x item matrix, item-factor kNN

Caveat (recorded): scoring aggregates item factors over recent history rather
than a proper user-vector fold-in, so ALS numbers are a floor, not a ceiling.
"""
import argparse
import os
import sys

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protocol import Bench, load_sessions  # noqa: E402

HIST_FOR_USER_VEC = 20


def user_item_run(bench, factors, alpha, weighted):
    from implicit.als import AlternatingLeastSquares
    rows, cols, vals = [], [], []
    uix = {u: k for k, u in enumerate(bench.train_by_user)}
    for u, h in bench.train_by_user.items():
        for i in h:
            w = 3.0 if (weighted and (u, i) in bench.strong) else 1.0
            rows.append(uix[u])
            cols.append(bench.item_ix[i])
            vals.append(w)
    UI = sparse.csr_matrix(
        (np.array(vals, np.float32), (rows, cols)),
        shape=(len(uix), bench.n_items))
    model = AlternatingLeastSquares(factors=factors, regularization=0.01,
                                    alpha=alpha, iterations=20, random_state=7)
    model.fit(UI, show_progress=False)
    IF, UF = model.item_factors, model.user_factors

    def fn(u):
        k = uix.get(u)
        if k is None:
            return None
        return (IF @ UF[k]).astype(np.float32)

    bench.evaluate(f'ALS-user f={factors} a={alpha} w={weighted}', fn)


def session_run(bench, factors, alpha):
    from implicit.als import AlternatingLeastSquares
    sessions = load_sessions(bench)
    sids = list(sessions)
    six = {s: k for k, s in enumerate(sids)}
    rows, cols = [], []
    for s, seq in sessions.items():
        for i in seq:
            j = bench.item_ix.get(i)
            if j is not None:
                rows.append(six[s])
                cols.append(j)
    SI = sparse.csr_matrix(
        (np.ones(len(rows), np.float32), (rows, cols)),
        shape=(len(sids), bench.n_items))
    model = AlternatingLeastSquares(factors=factors, regularization=0.01,
                                    alpha=alpha, iterations=15, random_state=7)
    model.fit(SI, show_progress=False)
    IF = model.item_factors
    IFn = IF / np.maximum(np.linalg.norm(IF, axis=1, keepdims=True), 1e-9)

    def fn(u):
        hix = [bench.item_ix[i]
               for i in bench.train_by_user[u][-HIST_FOR_USER_VEC:]
               if i in bench.item_ix]
        if not hix:
            return None
        uv = IFn[hix].mean(axis=0)
        n = np.linalg.norm(uv)
        if n < 1e-9:
            return None
        return (IFn @ (uv / n)).astype(np.float32)

    bench.evaluate(f'ALS-sessions f={factors} a={alpha}', fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', action='store_true')
    args = ap.parse_args()

    bench = Bench()
    bench.describe()

    if args.sessions:
        for factors, alpha in ((128, 40), (128, 10)):
            session_run(bench, factors, alpha)
    else:
        for factors, alpha, weighted in (
                (64, 40, False), (128, 40, False),
                (128, 40, True), (128, 10, True)):
            user_item_run(bench, factors, alpha, weighted)


if __name__ == '__main__':
    main()
