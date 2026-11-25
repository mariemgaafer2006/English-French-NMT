# English → French Neural Machine Translation (LSTM encoder-decoder)

A sequence-to-sequence model trained **from scratch** (no pretrained weights) on IWSLT 2017 en-fr with
TensorFlow/Keras and a custom training loop.

## Project layout
```
train.py                  training loop + validation + checkpointing   (entry point)
evaluate.py               validation loss / perplexity + BLEU (greedy and beam search)
translate.py              command-line translation (greedy + beam search)
app.py                    Gradio web demo
English_French_NMT_Colab.ipynb   step-by-step Colab runner
nmt/config.py             hyper-parameters (dataclass; every field is a CLI flag of train.py)
nmt/data.py               dataset loader  -> HF DatasetDict {train, validation, test}
nmt/tokenizer.py          BPE tokenizers (HF `tokenizers`), English + French, 10k vocab each
nmt/batching.py           batch data generator (ids, <s>/</s>, padding, length-bucketed batches)
nmt/model.py              LSTM encoder, LSTM decoder, Seq2Seq wrapper, masked loss
nmt/decoding.py           batched greedy decoding + beam search (length-normalised)
nmt/inference.py          load a checkpoint and translate text
tokenizers/               en_bpe.json, fr_bpe.json (already trained on the training split)
```

## How it maps to the assignment
| Building unit | Where |
|---|---|
| Dataset loader (train + validation splits) | `nmt/data.py` – train = `train.tags.en-fr.*` (232,825 pairs), validation = dev2010 (890), test = tst2010-2015 (8,597) |
| Tokenizer | `nmt/tokenizer.py` – BPE, trained on the **training split only**, separate EN / FR, ids `<pad>=0 <unk>=1 <s>=2 </s>=3` |
| Batch data generator | `nmt/batching.py` – decoder input `<s> + fr`, target `fr + </s>`, source `en + </s>`, right-padding |
| LSTM encoder | `nmt/model.py: Encoder` – Embedding(mask) → LSTM, returns final (h, c) |
| LSTM decoder | `nmt/model.py: Decoder` – Embedding → LSTM initialised with the encoder state → Dense(vocab) |
| Training loop | `train.py` – Adam, masked cross-entropy, gradient clipping, `tf.function` train step |
| Validation | `train.py` – validation loss + perplexity after every epoch |
| Checkpoints | `checkpoints/best.weights.h5` (lowest val loss), `last.weights.h5`, `config.json`, `history.json` |
| Beam search / greedy | `nmt/decoding.py`, used by `translate.py`, `evaluate.py`, `app.py` |

## Run it
### Option A – Google Colab (recommended, needs a GPU)
1. Upload `English_French_NMT_Project.zip` and `en-fr.zip` to Drive folder `MyDrive/NMT`.
2. Open `English_French_NMT_Colab.ipynb`, set the runtime to **T4 GPU**, run the cells top to bottom.

### Option B – local machine / Colab terminal
```bash
pip install -r requirements.txt
mkdir -p data && cp /path/to/en-fr.zip data/en-fr.zip      # the zip is extracted automatically on first run

python train.py --epochs 1 --max_train_samples 5000        # 1-2 min smoke test: checks everything works
python train.py                                            # full run: 15 epochs
python train.py --resume                                   # continue after an interruption

python evaluate.py                                         # val loss/perplexity + BLEU (greedy and beam 5)
python translate.py --text "Hello, how are you?" "Where is the train station?"
python translate.py --interactive
python app.py                                              # web demo at http://127.0.0.1:7860
```
If `data/en-fr.zip` is missing, the loader falls back to downloading `iwslt2017-en-fr` from the HuggingFace
Hub (`--source hf`); this fallback depends on the Hub's current dataset format and was not tested here.

## Default hyper-parameters
batch 32 · embedding 256 · LSTM units 512 · 1 layer · dropout 0.2 · Adam lr 1e-3 · 15 epochs ·
gradient clipping (global norm 1.0) · BPE vocab 10,000 per language · training pairs capped at 60 BPE tokens
(3.8% dropped, 224,031 kept → 7,001 steps/epoch). The learning rate is halved when validation loss does not
improve, and training stops after 3 epochs without improvement (`--patience 0` disables early stopping).
All of these are flags, e.g. `python train.py --units 1024 --num_layers 2 --epochs 20`.

## Notes and limitations
* The encoder passes only its **final state** to the decoder (as the assignment specifies, no attention),
  so quality drops on long sentences. Expect modest BLEU; validation loss and the sample translations are
  the main signals. Adding attention would be the natural next step.
* `--resume` restores the weights, epoch counter and learning rate; Adam's moment estimates restart from zero.
* Content is TED talks, so the model translates talk-style English best.
* Greedy decoding is batched (fast); beam search runs sentence by sentence (default width 5, length penalty α = 0.6).
