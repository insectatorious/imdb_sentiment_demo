# -*- coding: utf-8 -*-
import io
import json
import os
import logging
import argparse

import numpy as np
import tensorflow as tf
from tensorflow_datasets.core.features.text import SubwordTextEncoder

from attention import AttentionWeightedAverage
from utils import encode_text_with_encoder, preprocess_text, add_model_argparse

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def main(FLAGS) -> None:
  logging.info(f"Loading model from {FLAGS.saved_model}")
  model: tf.keras.Model = tf.keras.models.load_model(
    FLAGS.saved_model,
    custom_objects={"AttentionWeightedAverage": AttentionWeightedAverage}
  )
  logging.info(f"Loading encoder from {FLAGS.saved_encoder}")
  encoder_filename: str = FLAGS.saved_encoder.replace(".subwords", '')
  encoder: SubwordTextEncoder = SubwordTextEncoder.load_from_file(encoder_filename)

  sentence: str = preprocess_text(FLAGS.sentence)
  model_input = encode_text_with_encoder(encoder=encoder,
                                         text=sentence,
                                         max_sequence_length=FLAGS.max_words)
  pred: np.ndarray = np.squeeze(model.predict(model_input.reshape(1, -1)))
  logging.debug(f"Predictions: {pred}")

  pred_class: str = "Negative" if np.round(pred) == 0. else "Positive"
  logging.info(f"Classification is '{pred_class}' with a score of {pred}")

  if FLAGS.save_plots:
    logging.info("Visualising classification, saving to 'visualisations' dir")
    os.makedirs("visualisations", exist_ok=True)

    out_model = tf.keras.models.Model(inputs=model.input,
                                      outputs=(model.get_layer("attention").output[1],
                                               model.layers[-1].output))

    attention, posterior = out_model.predict(model_input.reshape(1, -1))

    with io.open(os.path.join("visualisations",
                              "neat_vision_single_string.json"),
                 "w",
                 encoding="utf-8") as m:
      tokens = [encoder.decode([idx]) for idx in encoder.encode(sentence)]
      nest_dict = dict(text=tokens,
                     label=int(np.round(pred)),
                     prediction=int(np.round(pred)),
                     # posterior=posterior[0].tolist(),
                     attention=attention[0].tolist(),
                     id=FLAGS.sentence[:16])
      json.dump([nest_dict], m)

    with io.open(os.path.join("visualisations",
                              "neat_vision_labels.json"),
                 "w",
                 encoding="utf-8") as m:
      json.dump({"0": {"name": "Negative", "desc": "Negative"},
                 "1": {"name": "Positive", "desc": "Positive"}},
                m)
  #   vis_model = tf.keras.models.Model(
  #     inputs=model.input,
  #     outputs=[layer.output for layer in model.layers])
  #
  #   feature_maps = vis_model.predict(model_input)
  #   layer_names_with_index: List[Tuple[int, str]] = [
  #     (index, layer.name)
  #     for index, layer in enumerate(model.layers)
  #     if "dropout" not in layer.name]
  #
  #   for i, layer_name in layer_names_with_index:
  #     save_feature_map(fig=visualise_feature_maps(feature_maps[i],
  #                                                 layer_name),
  #                      output_dir=FLAGS.plot_dir,
  #                      fname=f"{layer_name}.png")
  #
  #   # fig = visualise_feature_maps(feature_maps[1], layer_names_with_index[1][1])
  #   # fig.savefig(fname=os.path.join(FLAGS.plot_dir, "fmap1.png"))
  #
  #   cam_layer_with_index: Tuple[int, str] = [(i, name)
  #                for i, name in layer_names_with_index
  #                if "global_average_pooling" in name or "flatten" in name][-1]
  #   if cam_layer_with_index:
  #     logging.info(f"Class Activation Map Layer: {cam_layer_with_index[1]}")
  #     # We want the input to the layer
  #     cam_layer_with_index = cam_layer_with_index[0] - 1, cam_layer_with_index[1]
  #
  #     cam = get_cam(image_size=model_input.shape[1],
  #                   conv_out=feature_maps[cam_layer_with_index[0]],
  #                   pred_vec=pred,
  #                   all_amp_layer_weights=model.layers[-1].get_weights()[0],
  #                   filters=model.layers[cam_layer_with_index[0]].output.shape[-1])
  #     fig = plot_cam(model_input, cam)
  #     save_feature_map(fig, output_dir=FLAGS.plot_dir, fname=f"cam.png")
  #
  #   FLAGS.test_image.save(os.path.join(FLAGS.plot_dir, "input_image.png"))
  #   greyscale_image.save(os.path.join(FLAGS.plot_dir, "greyscale_input.png"))
  #   rescaled_image.save(os.path.join(FLAGS.plot_dir, "rescaled_model_input.png"))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser = add_model_argparse(parser)
  parser.add_argument(
    "sentence",
    default="",
    type=str,
    help="String sentence to classify"
  )
  parser.add_argument(
    "--save_plots",
    help="Save visualisations of this classification to "
         "'visualisations' directory",
    action="store_true")
  parser.add_argument(
    "--plot_dir",
    default="visualisations",
    type=str,
    help="Path to save visualisations to (dir)")

  FLAGS, unparsed = parser.parse_known_args()
  if unparsed:
    logging.warning("Unparsed arguments: {}".format(unparsed))

  logging.info("Arguments: {}".format(FLAGS))
  main(FLAGS)
