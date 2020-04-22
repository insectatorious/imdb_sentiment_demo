# -*- coding: utf-8 -*-
import io
import json
import logging
import argparse
import os
from typing import List, Any

import numpy as np
import tensorflow as tf

from utils import add_model_argparse, load_model_and_params, load_imdb_data, get_label_from_labelled_sample, \
  numeric_label_to_text, get_text_from_labelled_sample

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def main(FLAGS) -> None:
  batch_size = 10
  os.makedirs("tensorboard_assets", exist_ok=True)
  _, test_data, _ = load_imdb_data(max_words=FLAGS.max_words,
                                   batch_size=batch_size)

  model, encoder = load_model_and_params(FLAGS.saved_model,
                                         FLAGS.saved_encoder)

  test_data = test_data.shuffle(1000).take(200)
  reviews, labels, numeric_labels = [], [], []
  for batch_reviews, batch_labels in test_data.as_numpy_iterator():
    for i in range(batch_size):
      reviews.append(encoder.decode(batch_reviews[i]))
      labels.append(numeric_label_to_text(batch_labels[i]))
      numeric_labels.append(batch_labels[i])

  out_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[model.get_layer("attention").output,
             model.layers[-1].output])

  embedding, attention, pred = out_model.predict(test_data)
  print("@@@", embedding.shape, attention.shape, pred.shape)
  pred_class = [numeric_label_to_text(pred_numeric)[0]
                for pred_numeric in np.round(pred)]
  scores = np.squeeze(pred)

  np.savetxt(os.path.join("tensorboard_assets", "embedding_tensors.tsv"),
             X=embedding, delimiter="\t", encoding="utf-8")
  np.savetxt(os.path.join("tensorboard_assets", "attention_tensors.tsv"),
             X=attention, delimiter="\t", encoding="utf-8")
  with io.open(os.path.join("tensorboard_assets", "metadata.tsv"),
               "w", encoding='utf-8') as m:
    m.write("\t".join(["Label", "Predicted", "Score", "Text"]) + "\n")
    for label, pred, score, sentence in zip(labels,
                                            pred_class,
                                            scores,
                                            reviews):
      m.write("\t".join([str(label), str(pred), str(score), str(sentence)]) + "\n")

  with io.open("neat_vision.json", "w", encoding="utf-8") as m:
    nest_data = []
    for label, pred, attention, posterior, sentence in zip(numeric_labels,
                                                           np.round(scores),
                                                           attention,
                                                           scores,
                                                           reviews):
      tokens = [encoder.decode([idx]) for idx in encoder.encode(sentence)]
      nest_dict = dict(text=tokens,
                       label=int(label),
                       prediction=int(pred),
                       posterior=posterior.tolist(),
                       attention=attention.tolist(),
                       id=sentence[:16])
      nest_data.append(nest_dict)

    json.dump(nest_data, m)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser = add_model_argparse(parser)

  FLAGS, unparsed = parser.parse_known_args()
  if unparsed:
    logging.warning("Unparsed arguments: {}".format(unparsed))

  logging.info("Arguments: {}".format(FLAGS))
  main(FLAGS)
