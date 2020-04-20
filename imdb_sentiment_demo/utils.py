# -*- coding: utf-8 -*-
import os
import re
import argparse
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow_datasets.core.features.text.text_encoder import TextEncoder


def preprocess_text(text: str) -> str:
  if isinstance(text, bytes):
    text = text.decode("utf-8")
  text = text.lower()
  text = re.sub("[0-9]+", "NUM", text)
  text = re.sub('[^A-Za-z ]+', '', text)

  return text


def tf_preprocess_text(text):
  text_shape = text.shape
  [text, ] = tf.py_function(preprocess_text, [text], [tf.string])
  text.set_shape(text_shape)

  return text


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


def valid_path(path: str) -> str:
  if os.path.exists(path):
    if os.path.isdir(path):
      raise argparse.ArgumentTypeError(f"Expected '{path}' to be a file, got a folder")
  else:
    raise argparse.ArgumentTypeError(f"Expected '{path}' to be a valid path")

  return path


def add_model_argparse(parser: ArgumentParser) -> ArgumentParser:
  parser.add_argument(
    "--saved_model",
    default="model.h5",
    type=valid_path,
    help="Path to a saved TF Keras model")
  parser.add_argument(
    "--saved_encoder",
    default="vocab_1024",
    type=valid_path,
    help="Path to saved encoder (tfds.features.text.TextEncoder subclass)"
  )
  parser.add_argument(
    "--max_words",
    default=100,
    type=int,
    help="Max number of words in a single input to the network - must match "
         "the value used during training"
  )

  return parser
