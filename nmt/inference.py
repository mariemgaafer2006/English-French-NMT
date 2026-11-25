"""Load a trained checkpoint and translate raw English text."""
import os

from .config import Config
from .decoding import SearchDecoder
from .model import build_model
from .tokenizer import decode, encode, get_tokenizers

MAX_SRC_TOKENS = 100


class Translator:
    def __init__(self, ckpt_dir="checkpoints", tok_dir=None, which="best"):
        cfg_path = os.path.join(ckpt_dir, "config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"{cfg_path} not found - train a model first (python train.py).")
        self.cfg = Config.load(cfg_path)
        self.en_tok, self.fr_tok = get_tokenizers(tok_dir or self.cfg.tok_dir)
        self.model = build_model(self.cfg, self.en_tok.get_vocab_size(), self.fr_tok.get_vocab_size())
        w = os.path.join(ckpt_dir, f"{which}.weights.h5")
        self.model.load_weights(w)
        print(f"[translator] loaded {w}")
        self.dec = SearchDecoder(self.model)

    def _src(self, texts):
        return [ids[:MAX_SRC_TOKENS] or [1] for ids in encode(self.en_tok, texts)]

    def translate(self, texts, beam_size=1, alpha=0.6):
        """beam_size=1 -> greedy (batched); beam_size>1 -> beam search. texts: str or list of str."""
        single = isinstance(texts, str)
        texts = [texts] if single else texts
        src = self._src(texts)
        if beam_size <= 1:
            out = self.dec.greedy(src)
        else:
            out = [self.dec.beam(s, beam_size=beam_size, alpha=alpha) for s in src]
        res = [decode(self.fr_tok, ids) for ids in out]
        return res[0] if single else res
