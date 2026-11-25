"""Dataset loader: IWSLT 2017 English-French -> HuggingFace DatasetDict (train / validation / test).

The returned object has the same schema as ``load_dataset("iwslt2017", "iwslt2017-en-fr")``:
    ds["train"][i] == {"translation": {"en": "...", "fr": "..."}}

Two sources are supported:
  * local : the raw IWSLT17 files (en-fr.zip or an extracted folder) - the default when they exist
  * hf    : downloaded from the HuggingFace Hub with the ``datasets`` package
Splits: train = train.tags.en-fr.*, validation = dev2010, test = tst2010..tst2015.
"""
import html
import os
import re
import zipfile

from datasets import Dataset, DatasetDict

_SEG = re.compile(r"<seg id=\"\d+\">(.*?)</seg>", re.S)
_TEST_YEARS = [2010, 2011, 2012, 2013, 2014, 2015]


def _clean(s):
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def _read_train(path):
    """Training text: one or more sentences per line; lines that are XML tags (<doc>, <url> ...) are metadata."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("<") and s.endswith(">"):
                continue
            out.append(_clean(s))
    return out


def _read_xml(path):
    with open(path, encoding="utf-8") as f:
        return [_clean(m) for m in _SEG.findall(f.read())]


def _pairs(en, fr, name):
    assert len(en) == len(fr), f"{name}: {len(en)} English vs {len(fr)} French lines"
    keep = [(e, f) for e, f in zip(en, fr) if e and f]
    return Dataset.from_dict({"translation": [{"en": e, "fr": f} for e, f in keep]})


def _find_folder(data_path):
    """Return the folder that contains train.tags.en-fr.en (extracting the zip if needed)."""
    candidates = [data_path, os.path.join(data_path, "en-fr")]
    if data_path.endswith(".zip") and os.path.isfile(data_path):
        out = os.path.splitext(data_path)[0] + "_extracted"
        if not os.path.isdir(out):
            print(f"[data] extracting {data_path} -> {out}")
            with zipfile.ZipFile(data_path) as z:
                z.extractall(out)
        candidates = [out, os.path.join(out, "en-fr")]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "train.tags.en-fr.en")):
            return c
    return None


def load_local(data_path):
    folder = _find_folder(data_path)
    if folder is None:
        raise FileNotFoundError(
            f"Could not find the raw IWSLT17 files in '{data_path}'. Put en-fr.zip at data/en-fr.zip "
            "(or pass --data_path), or use --source hf.")
    p = lambda n: os.path.join(folder, n)
    train = _pairs(_read_train(p("train.tags.en-fr.en")), _read_train(p("train.tags.en-fr.fr")), "train")
    val = _pairs(_read_xml(p("IWSLT17.TED.dev2010.en-fr.en.xml")),
                 _read_xml(p("IWSLT17.TED.dev2010.en-fr.fr.xml")), "validation")
    te_en, te_fr = [], []
    for y in _TEST_YEARS:
        te_en += _read_xml(p(f"IWSLT17.TED.tst{y}.en-fr.en.xml"))
        te_fr += _read_xml(p(f"IWSLT17.TED.tst{y}.en-fr.fr.xml"))
    test = _pairs(te_en, te_fr, "test")
    return DatasetDict({"train": train, "validation": val, "test": test})


def load_hf():
    from datasets import load_dataset
    last = None
    for repo in ("IWSLT/iwslt2017", "iwslt2017"):
        try:
            return load_dataset(repo, "iwslt2017-en-fr")
        except Exception as e:  # newer `datasets` versions no longer run dataset scripts
            last = e
    raise RuntimeError(f"Could not load iwslt2017-en-fr from the Hub ({last}). Use the local en-fr.zip instead.")


def load_iwslt(data_path="data/en-fr.zip", source="auto"):
    if source == "hf":
        ds = load_hf()
    elif source == "local":
        ds = load_local(data_path)
    else:  # auto: local files if present, otherwise the Hub
        try:
            ds = load_local(data_path)
        except FileNotFoundError as e:
            print(f"[data] {e}\n[data] falling back to the HuggingFace Hub ...")
            ds = load_hf()
    print("[data] " + ", ".join(f"{k}: {len(v):,}" for k, v in ds.items()))
    return ds


def split_pairs(split):
    """HF split -> (english_list, french_list)."""
    tr = split["translation"]
    return [t["en"] for t in tr], [t["fr"] for t in tr]
