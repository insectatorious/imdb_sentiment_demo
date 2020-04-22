# -*- coding: utf-8 -*-
import logging
import os
import re
import argparse
from argparse import ArgumentParser
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow_datasets.core.features.text import SubwordTextEncoder
from tensorflow_datasets.core.features.text.text_encoder import TextEncoder

from attention import AttentionWeightedAverage


def numeric_label_to_text(label_numeric):
  if isinstance(label_numeric, np.ndarray):
    if len(label_numeric) == 0:
      return []
    return [numeric_label_to_text(label_numeric[0])] + [numeric_label_to_text(label_numeric[1:])]
  return "Negative" if label_numeric == 0. else "Positive"


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


def load_model_and_params(model_path: str, encoder_path: str) -> Tuple:
  logging.info(f"Loading model from {model_path}")
  model: tf.keras.Model = tf.keras.models.load_model(
    model_path,
    custom_objects={"AttentionWeightedAverage": AttentionWeightedAverage}
  )
  logging.info(f"Loading encoder from {encoder_path}")
  encoder_filename: str = encoder_path.replace(".subwords", '')
  encoder: SubwordTextEncoder = SubwordTextEncoder.load_from_file(encoder_filename)

  return model, encoder


def get_text_from_labelled_sample(text, _): return text


def get_label_from_labelled_sample(_, label): return label


def load_imdb_data(batch_size: int = 32,
                   max_words: int = 500,
                   vocab_size: int = 2**15,
                   buffer_size: int = 10000) -> Tuple:
  (train, test), info = tfds.load("imdb_reviews",
                                  as_supervised=True,
                                  with_info=True,
                                  split=["train", "test"])

  sentences = train.map(get_text_from_labelled_sample)
  sentences = sentences.map(tf_preprocess_text)

  vocab_file = f"vocab_{vocab_size}"
  logging.info(vocab_file)
  if tf.io.gfile.exists(f"{vocab_file}.subwords"):
    logging.info(f"Existing vocab file found at {vocab_file}.subwords, loading")
    encoder = SubwordTextEncoder.load_from_file(vocab_file)
  else:
    logging.info(f"No vocab file found at {vocab_file}.subwords, building")
    encoder = SubwordTextEncoder.build_from_corpus(sentences.as_numpy_iterator(),
                                                   vocab_size)
    encoder.save_to_file(vocab_file)
    logging.info(f"Vocab file saved at {vocab_file}.subwords")

  def encode(text_tensor, label):
    return encode_text_with_encoder(encoder,
                                    preprocess_text(text_tensor.numpy()),
                                    max_words), label

  def tf_encode(text, label):
    return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))

  train_data = train.map(tf_encode)
  train_data = train_data.shuffle(buffer_size)
  train_data = train_data.padded_batch(batch_size, padded_shapes=([-1], []))

  test_data = test.map(tf_encode)
  test_data = test_data.padded_batch(batch_size, padded_shapes=([-1], []))

  return train_data, test_data, encoder
