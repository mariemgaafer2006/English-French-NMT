"""Validation loss / perplexity and BLEU (sacreBLEU) of a trained checkpoint.

    python evaluate.py                       # validation loss + BLEU on validation & test (first 1000 sentences each)
    python evaluate.py --max_sents 0 --beam 5   # whole sets, also with beam search (slower)
"""
import argparse
import json
import os
import time

import numpy as np
import sacrebleu
import tensorflow as tf

from nmt.batching import BatchGenerator, encode_pairs
from nmt.data import load_iwslt, split_pairs
from nmt.inference import Translator
from nmt.model import masked_loss

p = argparse.ArgumentParser()
p.add_argument("--ckpt_dir", default="checkpoints")
p.add_argument("--which", default="best", choices=["best", "last"])
p.add_argument("--max_sents", type=int, default=1000, help="sentences per set for BLEU (0 = all)")
p.add_argument("--beam", type=int, default=5, help="also report beam search with this width (<=1 to skip)")
p.add_argument("--data_path", default=None)
a = p.parse_args()

tr = Translator(a.ckpt_dir, which=a.which)
cfg = tr.cfg
ds = load_iwslt(a.data_path or cfg.data_path, cfg.source)
results = {}

# --- validation loss -------------------------------------------------------------------------
va_en, va_fr = split_pairs(ds["validation"])
pairs, _ = encode_pairs(tr.en_tok, tr.fr_tok, va_en, va_fr, cfg.max_len)
tot = cnt = 0.0
for src, di, do in BatchGenerator(pairs, 128, shuffle=False):
    s, n = masked_loss(tf.constant(do), tr.model(tf.constant(src), tf.constant(di), training=False))
    tot += float(s); cnt += float(n)
results["validation_loss"], results["validation_ppl"] = tot / cnt, float(np.exp(tot / cnt))
print(f"validation loss {tot/cnt:.4f}  perplexity {np.exp(tot/cnt):.2f}")

# --- BLEU --------------------------------------------------------------------------------------
for split in ("validation", "test"):
    en, fr = split_pairs(ds[split])
    if a.max_sents:
        en, fr = en[:a.max_sents], fr[:a.max_sents]
    for name, k in [("greedy", 1)] + ([(f"beam{a.beam}", a.beam)] if a.beam > 1 else []):
        t0 = time.time()
        hyp = []
        for i in range(0, len(en), 64):
            hyp += tr.translate(en[i:i + 64], k)
        bleu = sacrebleu.corpus_bleu(hyp, [fr]).score
        results[f"bleu_{split}_{name}"] = bleu
        print(f"BLEU {split:10s} {name:7s} {bleu:5.2f}   ({len(en)} sentences, {time.time()-t0:.0f}s)")

os.makedirs(cfg.results_dir, exist_ok=True)
json.dump(results, open(os.path.join(cfg.results_dir, "metrics.json"), "w"), indent=2)
