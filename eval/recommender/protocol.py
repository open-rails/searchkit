"""The judge: shared eval protocol every baseline and candidate imports.

Consumes a normalized interaction log (see README input contract) from
RECOMMENDER_DATA (default: ./data next to this file):

  interactions.csv  user_id,item_id,timestamp,weight

Protocol: global TEMPORAL split (train = first `train_fraction` of time),
full-catalog ranking per warm user (seen-train excluded), recall@10/20 +
NDCG@10, head/tail stratification (head = top 10% items by train popularity).
"""
import csv
import collections
import math
import os
import sys

import numpy as np

DATA = os.environ.get(
    'RECOMMENDER_DATA',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
)
K_EVAL = (10, 20)


class Bench:
    """Loads the dataset, applies the temporal split, and judges score
    functions. A score function takes a user id and returns an np.float32
    array of length n_items (or None when it cannot score that user)."""

    def __init__(self, data_dir=DATA, train_fraction=0.8, strong_only=False):
        self.data_dir = data_dir
        first = {}
        strong = set()
        with open(os.path.join(data_dir, 'interactions.csv')) as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                if len(row) < 3:
                    continue
                w = float(row[3]) if len(row) > 3 and row[3] else 1.0
                if strong_only and w <= 1.0:
                    continue
                key = (row[0], row[1])
                if key not in first or row[2] < first[key]:
                    first[key] = row[2]
                if w > 1.0:
                    strong.add(key)
        self.strong = strong

        events = sorted(((ts, u, i) for (u, i), ts in first.items()))
        self.n_interactions = len(events)
        self.cut = events[int(len(events) * train_fraction)][0]
        self.train_by_user = collections.defaultdict(list)  # time-ordered
        self.test_by_user = collections.defaultdict(set)
        for ts, u, i in events:
            if ts < self.cut:
                self.train_by_user[u].append(i)
            else:
                self.test_by_user[u].add(i)

        self.item_pop = collections.Counter(
            i for h in self.train_by_user.values() for i in h)
        self.items = sorted(self.item_pop)
        self.item_ix = {it: k for k, it in enumerate(self.items)}
        self.n_items = len(self.items)

        self.eval_users = []
        for u, tset in self.test_by_user.items():
            hist = self.train_by_user.get(u)
            if not hist:
                continue
            truth = {i for i in tset
                     if i in self.item_ix and i not in set(hist)}
            if truth:
                self.eval_users.append((u, truth))
        self.cold_users = sum(
            1 for u in self.test_by_user if u not in self.train_by_user)

        head_cutoff = sorted(self.item_pop.values(), reverse=True)[
            max(1, self.n_items // 10) - 1]
        self.is_head = {i: self.item_pop[i] >= head_cutoff for i in self.items}

        self.pop_scores = np.array(
            [self.item_pop[i] for i in self.items], dtype=np.float32)

    def describe(self):
        print(f'bench: interactions={self.n_interactions:,} '
              f'catalog={self.n_items:,} eval_users={len(self.eval_users):,} '
              f'cold(excluded)={self.cold_users:,} cut={self.cut}')
        sys.stdout.flush()

    def evaluate(self, name, score_fn, adjacency=False, history_cap=20):
        """Judge one score function. Prints one line; returns the metrics.

        adjacency: also report the fraction of hits within +/-10 of a numeric
        history item id (the browse-adjacency audit; needs numeric ids)."""
        rec = {k: [] for k in K_EVAL}
        nd = []
        head_hits = head_n = tail_hits = tail_n = 0
        adj_hits = tot_hits = 0
        for u, truth in self.eval_users:
            scores = score_fn(u)
            if scores is None:
                scores = np.zeros(self.n_items, dtype=np.float32)
            seen = [self.item_ix[i] for i in set(self.train_by_user[u])
                    if i in self.item_ix]
            scores[seen] = -np.inf
            kmax = max(K_EVAL)
            top = np.argpartition(scores, -kmax)[-kmax:]
            top = top[np.argsort(scores[top])[::-1]]
            truth_ix = {self.item_ix[i] for i in truth}
            for k in K_EVAL:
                hits = len(truth_ix & set(top[:k].tolist()))
                rec[k].append(hits / min(len(truth_ix), k))
            dcg = sum(1 / math.log2(r + 2)
                      for r, j in enumerate(top[:10]) if j in truth_ix)
            idcg = sum(1 / math.log2(r + 2)
                       for r in range(min(10, len(truth_ix))))
            nd.append(dcg / idcg if idcg else 0)
            top20 = set(top[:20].tolist())
            for i in truth:
                if self.is_head[i]:
                    head_n += 1
                    head_hits += self.item_ix[i] in top20
                else:
                    tail_n += 1
                    tail_hits += self.item_ix[i] in top20
            if adjacency:
                hist_ids = [int(i) for i in self.train_by_user[u][-history_cap:]
                            if i.isdigit()]
                for j in top20:
                    if j in truth_ix:
                        tot_hits += 1
                        iid = self.items[j]
                        if iid.isdigit() and hist_ids and \
                                min(abs(int(iid) - h) for h in hist_ids) <= 10:
                            adj_hits += 1
        out = {
            'recall@10': float(np.mean(rec[10])),
            'recall@20': float(np.mean(rec[20])),
            'ndcg@10': float(np.mean(nd)),
            'head_hit@20': head_hits / max(head_n, 1),
            'tail_hit@20': tail_hits / max(tail_n, 1),
        }
        extra = ''
        if adjacency and tot_hits:
            extra = (f' | id-adjacent(<=10) hits: '
                     f'{100 * adj_hits / tot_hits:.0f}% of {tot_hits}')
        print(f'{name:42} recall@10={out["recall@10"]:.3f} '
              f'recall@20={out["recall@20"]:.3f} ndcg@10={out["ndcg@10"]:.3f} '
              f'| head-hit@20={out["head_hit@20"]:.3f} '
              f'tail-hit@20={out["tail_hit@20"]:.3f} ({tail_n} tail){extra}')
        sys.stdout.flush()
        return out


def topk_sparsify(S, k):
    """Keep each CSR row's k largest values (in place semantics)."""
    for r in range(S.shape[0]):
        lo, hi = S.indptr[r], S.indptr[r + 1]
        if hi - lo > k:
            idx = np.argpartition(S.data[lo:hi], -k)[-k:]
            keep = np.zeros(hi - lo, dtype=bool)
            keep[idx] = True
            S.data[lo:hi][~keep] = 0
    S.eliminate_zeros()
    return S


def load_sessions(bench, path=None):
    """Session sequences (session_id -> ordered items) truncated to events
    strictly before the bench's temporal cut (leakage guard)."""
    path = path or os.path.join(bench.data_dir, 'sessions.csv')
    sessions = collections.defaultdict(list)
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) < 3 or row[2] >= bench.cut:
                continue
            sessions[row[0]].append(row[1])
    return sessions
