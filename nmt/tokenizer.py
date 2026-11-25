"""BPE tokenizers (HuggingFace `tokenizers`), one for English and one for French.

Metaspace pre-tokenizer + Metaspace decoder (SentencePiece style) so that ``decode`` restores
normal spacing. Special ids: <pad>=0 <unk>=1 <s>=2 </s>=3 (the same in both languages).
"""
import os

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from .config import SPECIALS


def _metaspace():
    try:
        return pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always"), \
               decoders.Metaspace(replacement="▁", prepend_scheme="always")
    except TypeError:  # older `tokenizers`
        return pre_tokenizers.Metaspace(replacement="▁", add_prefix_space=True), \
               decoders.Metaspace(replacement="▁", add_prefix_space=True)


def train_bpe(sentences, vocab_size=10000):
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.normalizer = normalizers.NFKC()
    pre, dec = _metaspace()
    tok.pre_tokenizer, tok.decoder = pre, dec
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS, show_progress=False)
    tok.train_from_iterator(sentences, trainer=trainer)
    for i, s in enumerate(SPECIALS):
        assert tok.token_to_id(s) == i, "special tokens must occupy ids 0..3"
    return tok


def get_tokenizers(tok_dir, en_sentences=None, fr_sentences=None, vocab_size=10000, force=False):
    """Load tokenizers from `tok_dir`, or train and save them if they do not exist."""
    en_path, fr_path = os.path.join(tok_dir, "en_bpe.json"), os.path.join(tok_dir, "fr_bpe.json")
    if os.path.exists(en_path) and os.path.exists(fr_path) and not force:
        return Tokenizer.from_file(en_path), Tokenizer.from_file(fr_path)
    assert en_sentences is not None and fr_sentences is not None, "no tokenizers on disk and no data to train them"
    os.makedirs(tok_dir, exist_ok=True)
    print(f"[tokenizer] training BPE tokenizers (vocab {vocab_size}) ...")
    en, fr = train_bpe(en_sentences, vocab_size), train_bpe(fr_sentences, vocab_size)
    en.save(en_path)
    fr.save(fr_path)
    return en, fr


def encode(tok, sentences):
    return [e.ids for e in tok.encode_batch(sentences)]


def decode(tok, ids):
    return tok.decode(list(ids), skip_special_tokens=True)
