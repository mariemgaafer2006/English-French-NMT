"""Batch data generator: sentence pairs -> token ids -> padded numpy batches.

For a pair (en, fr) we build
    encoder input : en_ids + [</s>]
    decoder input : [<s>] + fr_ids          (teacher forcing)
    decoder target: fr_ids + [</s>]
Sequences are right-padded with 0 (<pad>). Batches are formed from length-sorted pools so that
padding is small, then shuffled.
"""
import numpy as np

from .config import BOS, EOS, PAD
from .tokenizer import encode


def encode_pairs(en_tok, fr_tok, en, fr, max_len=None, max_ratio=None):
    """Return a list of (src_ids, tgt_ids) with EOS/BOS NOT yet added. Optionally filters bad pairs."""
    src, tgt = encode(en_tok, en), encode(fr_tok, fr)
    pairs, dropped = [], 0
    for s, t in zip(src, tgt):
        if not s or not t:
            dropped += 1
            continue
        if max_len and (len(s) > max_len or len(t) > max_len):
            dropped += 1
            continue
        if max_ratio and max(len(s), len(t)) > 5 and max(len(s), len(t)) / max(1, min(len(s), len(t))) > max_ratio:
            dropped += 1
            continue
        pairs.append((s, t))
    return pairs, dropped


def pad(seqs):
    n = max(len(s) for s in seqs)
    out = np.full((len(seqs), n), PAD, dtype=np.int32)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out


def make_batch(pairs):
    src = pad([s + [EOS] for s, _ in pairs])
    dec_in = pad([[BOS] + t for _, t in pairs])
    dec_out = pad([t + [EOS] for _, t in pairs])
    return src, dec_in, dec_out


class BatchGenerator:
    """Iterable of (src, dec_in, dec_out) numpy batches. Re-iterating gives a new shuffle if shuffle=True."""

    def __init__(self, pairs, batch_size, shuffle=True, seed=0, pool_batches=50):
        self.pairs, self.bs, self.shuffle = pairs, batch_size, shuffle
        self.pool = batch_size * pool_batches
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return (len(self.pairs) + self.bs - 1) // self.bs

    def __iter__(self):
        idx = np.arange(len(self.pairs))
        if self.shuffle:
            self.rng.shuffle(idx)
        batches = []
        for start in range(0, len(idx), self.pool):
            chunk = sorted(idx[start:start + self.pool], key=lambda i: len(self.pairs[i][0]) + len(self.pairs[i][1]))
            batches += [chunk[i:i + self.bs] for i in range(0, len(chunk), self.bs)]
        if self.shuffle:
            self.rng.shuffle(batches)
        for b in batches:
            yield make_batch([self.pairs[i] for i in b])
