"""Train the English->French LSTM encoder-decoder.

    python train.py                                   # full run with the default hyper-parameters
    python train.py --epochs 1 --max_train_samples 5000   # quick smoke test
    python train.py --resume                          # continue from checkpoints/last.weights.h5
"""
import argparse
import dataclasses
import json
import os
import time

import numpy as np
import tensorflow as tf

from nmt.batching import BatchGenerator, encode_pairs
from nmt.config import Config
from nmt.data import load_iwslt, split_pairs
from nmt.inference import Translator
from nmt.model import build_model, masked_loss
from nmt.tokenizer import get_tokenizers

SAMPLES = [
    "Hello, how are you?",
    "I would like a cup of coffee, please.",
    "Where is the nearest train station?",
    "Education is the most powerful weapon which you can use to change the world.",
    "We are going to talk about climate change and what we can do about it.",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in dataclasses.fields(Config):
        p.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    p.add_argument("--resume", action="store_true", help="continue from <ckpt_dir>/last.weights.h5")
    p.add_argument("--retrain_tokenizers", action="store_true")
    a = vars(p.parse_args())
    resume, retrain = a.pop("resume"), a.pop("retrain_tokenizers")
    return Config(**a), resume, retrain


def main():
    cfg, resume, retrain = parse_args()
    tf.keras.utils.set_random_seed(cfg.seed)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)
    print("GPUs:", tf.config.list_physical_devices("GPU") or "none (training on CPU will be very slow)")

    # ---- 1. dataset -------------------------------------------------------------------------
    ds = load_iwslt(cfg.data_path, cfg.source)
    tr_en, tr_fr = split_pairs(ds["train"])
    va_en, va_fr = split_pairs(ds["validation"])
    if cfg.max_train_samples:
        tr_en, tr_fr = tr_en[:cfg.max_train_samples], tr_fr[:cfg.max_train_samples]

    # ---- 2. tokenizers (trained on the training split only) ----------------------------------
    en_tok, fr_tok = get_tokenizers(cfg.tok_dir, tr_en, tr_fr, cfg.vocab_size, force=retrain)
    print(f"[tokenizer] vocab: en={en_tok.get_vocab_size()} fr={fr_tok.get_vocab_size()}")

    # ---- 3. batch generators -----------------------------------------------------------------
    train_pairs, d1 = encode_pairs(en_tok, fr_tok, tr_en, tr_fr, cfg.max_len, cfg.max_ratio)
    val_pairs, d2 = encode_pairs(en_tok, fr_tok, va_en, va_fr, cfg.max_len, None)
    print(f"[data] train pairs: {len(train_pairs):,} (dropped {d1:,}) | validation pairs: {len(val_pairs):,} (dropped {d2:,})")
    train_gen = BatchGenerator(train_pairs, cfg.batch_size, shuffle=True, seed=cfg.seed)
    val_gen = BatchGenerator(val_pairs, 128, shuffle=False)

    # ---- 4. model / optimizer ----------------------------------------------------------------
    model = build_model(cfg, en_tok.get_vocab_size(), fr_tok.get_vocab_size())
    n_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"[model] LSTM encoder-decoder, {n_params/1e6:.1f}M trainable parameters")
    opt = tf.keras.optimizers.Adam(cfg.lr)
    cfg.save(os.path.join(cfg.ckpt_dir, "config.json"))

    sig = [tf.TensorSpec([None, None], tf.int32)] * 3

    @tf.function(input_signature=sig)
    def train_step(src, dec_in, dec_out):
        with tf.GradientTape() as tape:
            logits = model(src, dec_in, training=True)
            loss_sum, n = masked_loss(dec_out, logits)
            loss = loss_sum / n
        grads = tape.gradient(loss, model.trainable_variables)
        if cfg.clipnorm:
            grads, _ = tf.clip_by_global_norm(grads, cfg.clipnorm)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss_sum, n

    @tf.function(input_signature=sig)
    def eval_step(src, dec_in, dec_out):
        return masked_loss(dec_out, model(src, dec_in, training=False))

    def validate():
        tot, cnt = 0.0, 0.0
        for src, di, do in val_gen:
            s, n = eval_step(src, di, do)
            tot += float(s); cnt += float(n)
        return tot / cnt

    # ---- 5. resume ---------------------------------------------------------------------------
    hist_path = os.path.join(cfg.ckpt_dir, "history.json")
    history, start_epoch, best, bad = [], 0, float("inf"), 0
    if resume and os.path.exists(os.path.join(cfg.ckpt_dir, "last.weights.h5")):
        model.load_weights(os.path.join(cfg.ckpt_dir, "last.weights.h5"))
        history = json.load(open(hist_path))
        start_epoch = len(history)
        best = min(h["val_loss"] for h in history)
        opt.learning_rate.assign(history[-1]["lr"])
        print(f"[resume] continuing after epoch {start_epoch} (best val loss {best:.4f}; Adam moments restart from zero)")

    # ---- 6. training loop --------------------------------------------------------------------
    for epoch in range(start_epoch, cfg.epochs):
        t0, tot, cnt = time.time(), 0.0, 0.0
        for step, (src, di, do) in enumerate(train_gen, 1):
            s, n = train_step(src, di, do)
            tot += float(s); cnt += float(n)
            if step % 500 == 0 or step == len(train_gen):
                print(f"  epoch {epoch+1}/{cfg.epochs} step {step}/{len(train_gen)} "
                      f"train loss {tot/cnt:.4f}  ({(time.time()-t0)/step*1000:.0f} ms/step)", flush=True)
        train_loss = tot / cnt
        val_loss = validate()
        lr = float(opt.learning_rate.numpy())
        rec = dict(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss, val_ppl=float(np.exp(val_loss)),
                   lr=lr, seconds=time.time() - t0)
        history.append(rec)
        print(f"== epoch {epoch+1}: train loss {train_loss:.4f} | val loss {val_loss:.4f} "
              f"(ppl {rec['val_ppl']:.1f}) | lr {lr:.5f} | {rec['seconds']/60:.1f} min")

        model.save_weights(os.path.join(cfg.ckpt_dir, "last.weights.h5"))
        if val_loss < best:
            best, bad = val_loss, 0
            model.save_weights(os.path.join(cfg.ckpt_dir, "best.weights.h5"))
            print("   saved new best checkpoint")
        else:
            bad += 1
            opt.learning_rate.assign(lr * cfg.lr_decay_factor)
            if cfg.patience and bad >= cfg.patience:
                print(f"[early stop] no validation improvement for {bad} epochs")
                json.dump(history, open(hist_path, "w"), indent=2)
                break
        json.dump(history, open(hist_path, "w"), indent=2)

    # ---- 7. curves + sample translations -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = [h["epoch"] for h in history]
        plt.figure(figsize=(6, 4))
        plt.plot(ep, [h["train_loss"] for h in history], marker="o", label="train")
        plt.plot(ep, [h["val_loss"] for h in history], marker="o", label="validation")
        plt.xlabel("epoch"); plt.ylabel("cross-entropy per token"); plt.legend(); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(cfg.results_dir, "loss_curve.png"), dpi=150)
    except Exception as e:
        print("[plot] skipped:", e)

    tr = Translator(cfg.ckpt_dir, cfg.tok_dir, "best")
    lines = []
    for s in SAMPLES:
        lines += [f"EN     : {s}", f"greedy : {tr.translate(s, 1)}", f"beam-5 : {tr.translate(s, 5)}", ""]
    open(os.path.join(cfg.results_dir, "sample_translations.txt"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    print("Done. Next: python evaluate.py   |   python translate.py --interactive   |   python app.py")


if __name__ == "__main__":
    main()
