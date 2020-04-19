# -*- coding: utf-8 -*-

import os
import logging
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow_datasets.core.features.text import SubwordTextEncoder

from attention import AttentionWeightedAverage

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


(train, test), info = tfds.load("imdb_reviews",
                                as_supervised=True,
                                with_info=True,
                                split=["train", "test"])


def get_text_from_labelled_sample(text, _): return text


def encode_text(text):
  encoded_text = encoder.encode(text)
  encoded_text = pad_sequences([encoded_text],
                               maxlen=1000,
                               padding="post",
                               truncating="post")
  encoded_text = np.squeeze(encoded_text)

  return encoded_text


def encode(text_tensor, label):
  return encode_text(text_tensor.numpy()), label


def encode_map_fn(text, label):
  return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))


BUFFER_SIZE: int = 15000
MAX_WORDS: int = 1000
BATCH_SIZE: int = 128
VOCAB_SIZE: int = 2 ** 15
VOCAB_FILE: str = f"vocab_{VOCAB_SIZE}"
EMBEDDING_DIM: int = 300
LSTM_CELLS: int = 25

sentences = train.map(get_text_from_labelled_sample)

if tf.io.gfile.exists(f"{VOCAB_FILE}.subwords"):
  logging.info(f"Existing vocab file found at {VOCAB_FILE}.subwords, loading")
  encoder = SubwordTextEncoder.load_from_file(VOCAB_FILE)
else:
  logging.info(f"No vocab file found at {VOCAB_FILE}.subwords, building")
  encoder = SubwordTextEncoder.build_from_corpus(sentences.as_numpy_iterator(),
                                                 VOCAB_SIZE)
  encoder.save_to_file(VOCAB_FILE)
  logging.info(f"Vocab file saved at {VOCAB_FILE}.subwords")

train_data = train.map(encode_map_fn)
train_data = train_data.shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))

test_data = test.map(encode_map_fn)
test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))


sample_text, sample_labels = next(iter(test_data))

logging.info(sample_text[0], sample_labels[0])


l2_scale = 1e-4
dropout_prob = 0.5

input_layer = tf.keras.layers.Input(shape=(MAX_WORDS,))
embedding_layer = tf.keras.layers.Embedding(encoder.vocab_size,
                                            EMBEDDING_DIM,
                                            mask_zero=True)(input_layer)
layer = tf.keras.layers.SpatialDropout1D(dropout_prob)(embedding_layer)
rnn_layer_list = [layer]
for i in range(5):
  layer = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(
      LSTM_CELLS,
      return_sequences=True,
      kernel_regularizer=tf.keras.regularizers.l2(l=l2_scale),
      recurrent_regularizer=tf.keras.regularizers.l2(l=l2_scale)
    )
  )(layer)
  layer = tf.keras.layers.SpatialDropout1D(dropout_prob)(layer)
  rnn_layer_list.append(layer)

layer = tf.keras.layers.concatenate(rnn_layer_list, name="rnn_concat")
layer = AttentionWeightedAverage(name="attention")(layer)
layer = tf.keras.layers.Dense(1, activation="sigmoid")(layer)

model = tf.keras.models.Model(inputs=[input_layer], outputs=[layer])

model.compile(tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

model.summary()
logging.info(f"Number of layers: {len(model.layers)}")

logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="model.h5",
                                                 verbose=1,
                                                 save_best_only=True)
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                write_images=True,
                                                profile_batch=0,
                                                update_freq="batch",
                                                histogram_freq=1)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                     min_delta=1e-2,
                                                     patience=4,
                                                     verbose=1,
                                                     restore_best_weights=True)


model.fit(train_data,
          epochs=30,
          validation_data=test_data,
          callbacks=[cp_callback, tensorboard_cb, early_stopping_cb])


tf.keras.utils.plot_model(model,
                          to_file=os.path.join(logdir, "model.png"),
                          show_shapes=True,
                          show_layer_names=False,
                          rankdir='TB',
                          expand_nested=False,
                          dpi=128)

logging.info(f"Final model evaluation")
model.evaluate(test_data)
