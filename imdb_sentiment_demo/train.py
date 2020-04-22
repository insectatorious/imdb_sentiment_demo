# -*- coding: utf-8 -*-

import os
import logging
from datetime import datetime

import tensorflow as tf

from attention import AttentionWeightedAverage
from utils import load_imdb_data

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


BUFFER_SIZE: int = 15000
MAX_WORDS: int = 500
BATCH_SIZE: int = 64
VOCAB_SIZE: int = 2 ** 15
VOCAB_FILE: str = f"vocab_{VOCAB_SIZE}"
EMBEDDING_DIM: int = 300
LSTM_CELLS: int = 300

train_data, test_data, encoder = load_imdb_data(batch_size=BATCH_SIZE,
                                                max_words=MAX_WORDS,
                                                vocab_size=VOCAB_SIZE,
                                                buffer_size=BUFFER_SIZE)

sample_text, sample_labels = next(iter(test_data))

logging.info(sample_text[0], sample_labels[0])


l2_scale = 1e-3
dropout_prob = 0.5

input_layer = tf.keras.layers.Input(shape=(MAX_WORDS,))
embedding_layer = tf.keras.layers.Embedding(encoder.vocab_size,
                                            EMBEDDING_DIM,
                                            mask_zero=True)(input_layer)
layer = tf.keras.layers.SpatialDropout1D(dropout_prob)(embedding_layer)
rnn_layer_list = [layer]
for i in range(2):
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
layer = AttentionWeightedAverage(name="attention", return_attention=True)(layer)
layer, weights = layer
layer = tf.keras.layers.Dense(1, activation="sigmoid")(layer)

model = tf.keras.models.Model(inputs=[input_layer], outputs=layer)

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
