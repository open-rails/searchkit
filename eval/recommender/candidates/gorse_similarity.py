"""Candidate: Gorse-style item-to-item similarity — IDF-weighted cosine over
user sets with significance shrinkage (the formula from gorse's
logics/item_to_item.go):

  similarity(a,b) = (commonSum * commonCount)
                    / (sqrt(Wa) * sqrt(Wb) * (commonCount + LAMBDA))
    commonSum   = sum of user-IDF weights of shared users
    commonCount = number of shared users
    Wa          = sum of user-IDF weights of a's users
    user IDF    = max(log(numItems / userFreq), 1e-3)  (damps power users)

  python candidates/gorse_similarity.py
"""
import os
import sys

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protocol import Bench, topk_sparsify  # noqa: E402


def build(bench, cap, lam, use_idf, topk=100):
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
    A.sum_duplicates()
    A.sort_indices()

    ufreq = np.asarray(A.sum(axis=1)).ravel()
    if use_idf:
        w = np.maximum(np.log(bench.n_items / np.maximum(ufreq, 1)),
                       1e-3).astype(np.float32)
    else:
        w = np.ones_like(ufreq, dtype=np.float32)

    C0 = (A.T @ A).tocsr()          # commonCount
    C0.sum_duplicates()
    C0.sort_indices()
    WA = sparse.diags(w) @ A
    C1 = (A.T @ WA).tocsr()         # commonSum
    C1.sum_duplicates()
    C1.sort_indices()
    assert np.array_equal(C0.indptr, C1.indptr)
    assert np.array_equal(C0.indices, C1.indices)

    Wa = np.asarray(WA.sum(axis=0)).ravel()
    sqrtWa = np.sqrt(np.maximum(Wa, 1e-9)).astype(np.float32)
    coo_rows = np.repeat(np.arange(bench.n_items), np.diff(C0.indptr))
    sim = (C1.data * C0.data) / (
        sqrtWa[coo_rows] * sqrtWa[C0.indices] * (C0.data + lam))
    S = sparse.csr_matrix(
        (sim.astype(np.float32), C0.indices.copy(), C0.indptr.copy()),
        shape=C0.shape)
    S = S.tolil()
    S.setdiag(0)
    S = S.tocsr()
    return topk_sparsify(S, topk)


def main():
    bench = Bench()
    bench.describe()

    for cap, lam, idf in ((20, 100, True), (20, 25, True), (20, 100, False),
                          (50, 100, True), (20, 300, True)):
        S = build(bench, cap=cap, lam=lam, use_idf=idf)

        def fn(u, S=S, cap=cap):
            hix = [bench.item_ix[i] for i in bench.train_by_user[u][-cap:]
                   if i in bench.item_ix]
            if not hix:
                return None
            return np.asarray(S[hix].sum(0)).ravel().astype(np.float32)

        bench.evaluate(f'gorse-sim cap={cap} lambda={lam} idf={idf}', fn)


if __name__ == '__main__':
    main()
