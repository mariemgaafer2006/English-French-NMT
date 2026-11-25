"""All hyper-parameters in one place. Every field can be overridden from the CLI of train.py."""
from dataclasses import dataclass, asdict
import json

PAD, UNK, BOS, EOS = 0, 1, 2, 3          # special token ids (same in both tokenizers)
SPECIALS = ["<pad>", "<unk>", "<s>", "</s>"]


@dataclass
class Config:
    # data
    data_path: str = "data/en-fr.zip"     # zip or folder with the raw IWSLT17 files
    source: str = "auto"                  # auto | local | hf  (hf = download from the HF Hub)
    max_len: int = 60                     # drop training pairs longer than this (in BPE tokens, either side)
    max_ratio: float = 3.0                # drop badly aligned pairs (length ratio), train only
    max_train_samples: int = 0            # 0 = all (use a small number for smoke tests)
    # tokenizer
    vocab_size: int = 10000
    tok_dir: str = "tokenizers"
    # model
    embed_dim: int = 256
    units: int = 512
    num_layers: int = 1
    dropout: float = 0.2
    # training
    batch_size: int = 32
    epochs: int = 15
    lr: float = 1e-3
    clipnorm: float = 1.0                 # global-norm gradient clipping, 0 = off
    lr_decay_factor: float = 0.5          # multiply lr by this when val loss does not improve
    patience: int = 3                     # early stopping (epochs without val improvement), 0 = off
    seed: int = 42
    ckpt_dir: str = "checkpoints"
    results_dir: str = "results"

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
