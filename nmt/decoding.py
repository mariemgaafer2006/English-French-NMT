"""Greedy (batched) and beam-search decoding for the LSTM Seq2Seq model."""
import numpy as np
import tensorflow as tf

from .batching import pad
from .config import BOS, EOS, PAD, UNK


class SearchDecoder:
    """Wraps a trained Seq2Seq model with compiled encode / step functions."""

    def __init__(self, model):
        self.model = model
        self._encode = tf.function(lambda s: model.encode(s), reduce_retracing=True)
        self._step = tf.function(lambda t, st: model.decode_step(t, st), reduce_retracing=True)

    # ---------------------------------------------------------------- greedy (whole batch at once)
    def greedy(self, src_ids, max_len=100):
        """src_ids: list of token-id lists (no EOS). Returns list of id lists (no BOS / EOS)."""
        B = len(src_ids)
        states = self._encode(tf.constant(pad([s + [EOS] for s in src_ids])))
        tokens = np.full((B, 1), BOS, dtype=np.int32)
        done = np.zeros(B, bool)
        out = [[] for _ in range(B)]
        longest = max(len(s) for s in src_ids)
        for _ in range(min(max_len, 2 * longest + 10)):
            logits, states = self._step(tf.constant(tokens), states)
            logits = logits.numpy()
            logits[:, [PAD, UNK, BOS]] = -1e9
            nxt = logits.argmax(-1)
            for i in range(B):
                if not done[i]:
                    if nxt[i] == EOS:
                        done[i] = True
                    else:
                        out[i].append(int(nxt[i]))
            if done.all():
                break
            tokens = np.where(done, EOS, nxt).astype(np.int32)[:, None]
        return out

    # ---------------------------------------------------------------- beam search (one sentence)
    def beam(self, src, beam_size=5, max_len=100, alpha=0.6):
        """src: list of token ids (no EOS). Length-normalised beam search (GNMT length penalty)."""
        states = self._encode(tf.constant(pad([src + [EOS]])))
        seqs, scores = [[BOS]], np.zeros(1)
        finished = []
        lp = lambda n: ((5.0 + n) / 6.0) ** alpha
        for t in range(min(max_len, 2 * len(src) + 10)):
            tokens = np.array([[s[-1]] for s in seqs], dtype=np.int32)
            logits, new_states = self._step(tf.constant(tokens), states)
            logits = logits.numpy().astype(np.float64)
            logp = logits - logits.max(-1, keepdims=True)
            logp = logp - np.log(np.exp(logp).sum(-1, keepdims=True))
            logp[:, [PAD, UNK, BOS]] = -np.inf
            if t == 0:
                logp[:, EOS] = -np.inf                       # no empty translations
            cand = (scores[:, None] + logp).ravel()
            V = logp.shape[1]
            order = np.argsort(-cand)[: 2 * beam_size]
            alive_seq, alive_score, parents = [], [], []
            for j in order:
                if not np.isfinite(cand[j]):
                    break
                b, w = divmod(int(j), V)
                if w == EOS:
                    finished.append((cand[j] / lp(t + 1), seqs[b][1:]))
                elif len(alive_seq) < beam_size:
                    alive_seq.append(seqs[b] + [w]); alive_score.append(cand[j]); parents.append(b)
            if not alive_seq or len(finished) >= beam_size:
                break
            seqs, scores = alive_seq, np.array(alive_score)
            idx = tf.constant(parents)
            states = tuple(tf.gather(s, idx) for s in new_states)
        if not finished:                                       # hit max_len: take the best unfinished one
            finished = [(scores[i] / lp(len(seqs[i]) - 1), seqs[i][1:]) for i in range(len(seqs))]
        return max(finished, key=lambda x: x[0])[1]
