# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow_datasets.core.features.text.text_encoder import TextEncoder


def encode_text_with_encoder(encoder: TextEncoder,
                             text: str,
                             max_sequence_length: int) -> np.ndarray:
  encoded_text = encoder.encode(text)
  encoded_text = pad_sequences([encoded_text],
                               maxlen=max_sequence_length,
                               padding="post",
                               truncating="post")
  encoded_text = np.squeeze(encoded_text)

  return encoded_text

