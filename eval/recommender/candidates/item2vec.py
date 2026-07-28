"""Candidate: item2vec (gensim skip-gram over interaction sequences), tuned
per Caselles-Dupre (ns_exponent matters far more than NLP defaults).

  python candidates/item2vec.py             # user-sequence corpus
  python candidates/item2vec.py --sessions  # session-sequence corpus
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protocol import Bench, load_sessions  # noqa: E402

HIST_FOR_USER_VEC = 20


def train_and_judge(bench, sentences, tag, ns_exponent, window, epochs, dims=100):
    from gensim.models import Word2Vec
    m = Word2Vec(sentences, vector_size=dims, window=window,
                 min_count=3 if tag == 'sessions' else 1, sg=1, negative=10,
                 ns_exponent=ns_exponent, sample=1e-4, workers=14,
                 epochs=epochs, seed=7)
    V = np.zeros((bench.n_items, dims), dtype=np.float32)
    cov = 0
    for i in bench.items:
        if i in m.wv:
            V[bench.item_ix[i]] = m.wv[i]
            cov += 1
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)

    def fn(u):
        hist = [i for i in bench.train_by_user[u][-HIST_FOR_USER_VEC:]
                if i in bench.item_ix]
        if not hist:
            return None
        uv = Vn[[bench.item_ix[i] for i in hist]].mean(axis=0)
        n = np.linalg.norm(uv)
        if n < 1e-9:
            return None
        return (Vn @ (uv / n)).astype(np.float32)

    bench.evaluate(
        f'item2vec-{tag} ns={ns_exponent:+.2f} win={window} (cov={cov})', fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', action='store_true',
                    help='train on session sequences instead of user sequences')
    ap.add_argument('--epochs', type=int, default=None)
    args = ap.parse_args()

    bench = Bench()
    bench.describe()

    if args.sessions:
        sessions = load_sessions(bench)
        sentences = [s for s in sessions.values() if len(s) >= 2]
        tag = 'sessions'
        epochs = args.epochs or 10
    else:
        sentences = [h for h in bench.train_by_user.values() if len(h) >= 2]
        tag = 'user-seq'
        epochs = args.epochs or 30
    print(f'corpus: {len(sentences):,} {tag} sequences')

    for ns in (0.75, 0.0, -0.5):
        for window in (5, 20) if not args.sessions else (5,):
            train_and_judge(bench, sentences, tag, ns, window, epochs)


if __name__ == '__main__':
    main()
