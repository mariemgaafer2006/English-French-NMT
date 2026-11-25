"""Translate English -> French with a trained checkpoint.

    python translate.py --text "Hello, how are you?"
    python translate.py --text "Good morning." "Thank you very much." --beam 5
    python translate.py --interactive
"""
import argparse

from nmt.inference import Translator

p = argparse.ArgumentParser()
p.add_argument("--text", nargs="*", default=[], help="one or more English sentences")
p.add_argument("--interactive", action="store_true")
p.add_argument("--beam", type=int, default=5, help="beam width (1 = greedy only)")
p.add_argument("--alpha", type=float, default=0.6, help="beam length-penalty strength")
p.add_argument("--ckpt_dir", default="checkpoints")
p.add_argument("--tok_dir", default=None)
p.add_argument("--which", default="best", choices=["best", "last"])
a = p.parse_args()

tr = Translator(a.ckpt_dir, a.tok_dir, a.which)


def show(s):
    print(f"EN     : {s}")
    print(f"greedy : {tr.translate(s, 1)}")
    if a.beam > 1:
        print(f"beam-{a.beam} : {tr.translate(s, a.beam, a.alpha)}")
    print()


for s in a.text:
    show(s)
if a.interactive or not a.text:
    print("Type an English sentence (empty line to quit).")
    while True:
        s = input("> ").strip()
        if not s:
            break
        show(s)
