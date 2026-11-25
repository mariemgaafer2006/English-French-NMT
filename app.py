"""Gradio demo:  python app.py            (add --share in Colab to get a public link)"""
import argparse

import gradio as gr

from nmt.inference import Translator

p = argparse.ArgumentParser()
p.add_argument("--ckpt_dir", default="checkpoints")
p.add_argument("--which", default="best", choices=["best", "last"])
p.add_argument("--share", action="store_true")
a = p.parse_args()

translator = Translator(a.ckpt_dir, which=a.which)


def run(text, method, beam_size):
    text = (text or "").strip()
    if not text:
        return ""
    k = 1 if method == "Greedy" else int(beam_size)
    # translate sentence by sentence: the model was trained on single sentences
    import re
    parts = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]
    return " ".join(translator.translate(parts, k))


demo = gr.Interface(
    fn=run,
    inputs=[gr.Textbox(lines=4, label="English", placeholder="Type an English sentence..."),
            gr.Radio(["Beam search", "Greedy"], value="Beam search", label="Decoding"),
            gr.Slider(2, 10, value=5, step=1, label="Beam width")],
    outputs=gr.Textbox(lines=4, label="French"),
    title="English -> French NMT (LSTM encoder-decoder)",
    examples=[["Hello, how are you?", "Beam search", 5],
              ["We are going to talk about climate change.", "Greedy", 5]],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(share=a.share)
