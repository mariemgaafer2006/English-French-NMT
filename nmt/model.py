"""LSTM encoder-decoder (sequence-to-sequence) in TensorFlow / Keras.

Encoder : Embedding(mask_zero) -> dropout -> LSTM x num_layers  ->  final (h, c) of every layer
Decoder : Embedding(mask_zero) -> dropout -> LSTM x num_layers (initialised with encoder states)
          -> dropout -> Dense(vocab)  -> logits for the next French token

States are passed around as a flat tuple (h1, c1, h2, c2, ...).
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import PAD


class Encoder(keras.Model):
    def __init__(self, vocab_size, embed_dim, units, num_layers=1, dropout=0.2, **kw):
        super().__init__(**kw)
        self.embedding = layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.drop = layers.Dropout(dropout)
        self.lstms = [layers.LSTM(units, return_sequences=True, return_state=True) for _ in range(num_layers)]

    def call(self, src, training=False):
        mask = tf.not_equal(src, PAD)
        x = self.drop(self.embedding(src), training=training)
        states = []
        for i, lstm in enumerate(self.lstms):
            x, h, c = lstm(x, mask=mask, training=training)   # h, c = state at the last real token
            if i < len(self.lstms) - 1:
                x = self.drop(x, training=training)
            states += [h, c]
        return tuple(states)


class Decoder(keras.Model):
    def __init__(self, vocab_size, embed_dim, units, num_layers=1, dropout=0.2, **kw):
        super().__init__(**kw)
        self.embedding = layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.drop = layers.Dropout(dropout)
        self.lstms = [layers.LSTM(units, return_sequences=True, return_state=True) for _ in range(num_layers)]
        self.out = layers.Dense(vocab_size)

    def call(self, dec_in, states, training=False):
        mask = tf.not_equal(dec_in, PAD)
        x = self.drop(self.embedding(dec_in), training=training)
        new_states = []
        for i, lstm in enumerate(self.lstms):
            x, h, c = lstm(x, mask=mask, initial_state=[states[2 * i], states[2 * i + 1]], training=training)
            if i < len(self.lstms) - 1:
                x = self.drop(x, training=training)
            new_states += [h, c]
        logits = self.out(self.drop(x, training=training))
        return logits, tuple(new_states)


class Seq2Seq(keras.Model):
    def __init__(self, src_vocab, tgt_vocab, embed_dim=256, units=512, num_layers=1, dropout=0.2, **kw):
        super().__init__(**kw)
        self.encoder = Encoder(src_vocab, embed_dim, units, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab, embed_dim, units, num_layers, dropout)

    def call(self, src, dec_in, training=False):
        """Teacher forcing: returns logits [batch, tgt_len, tgt_vocab]."""
        states = self.encoder(src, training=training)
        logits, _ = self.decoder(dec_in, states, training=training)
        return logits

    # --- helpers used at inference time -------------------------------------------------------
    def encode(self, src):
        return self.encoder(src, training=False)

    def decode_step(self, tokens, states):
        """tokens [batch, 1] -> (logits [batch, vocab], new_states)."""
        logits, states = self.decoder(tokens, states, training=False)
        return logits[:, -1, :], states


def build_model(cfg, src_vocab, tgt_vocab):
    model = Seq2Seq(src_vocab, tgt_vocab, cfg.embed_dim, cfg.units, cfg.num_layers, cfg.dropout)
    model(tf.zeros((2, 5), tf.int32) + 4, tf.zeros((2, 5), tf.int32) + 4)   # build all weights
    return model


def masked_loss(labels, logits):
    """Cross-entropy summed over real (non-pad) target tokens. Returns (loss_sum, token_count)."""
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    mask = tf.cast(tf.not_equal(labels, PAD), ce.dtype)
    return tf.reduce_sum(ce * mask), tf.reduce_sum(mask)
