"""
Created on March 14th 2019

@author: Sunhwan Lee

Chatbot using Joint Slot Filling and Intent Detection model implementation.
"""

import argparse
import os
import random
import sys

# import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

import utils
import inference
import train
import model_helper
import evaluation_utils
import vocab_utils
import nmt_utils

"""
python chatbot.py \
    --out_dir=output/rnn_intent/best_accuracy \
    --hparams_path=output/rnn_intent/best_accuracy

python chatbot.py \
    --out_dir=output/vanilla_seq2seq_joint/best_f1 \
    --hparams_path=output/vanilla_seq2seq_joint/best_f1
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

utils.check_tensorflow_version()

FLAGS = None

INFERENCE_KEYS = ["src_max_len_infer", "tgt_max_len_infer", "subword_option",
                  "infer_batch_size", "beam_width",
                  "length_penalty_weight", "coverage_penalty_weight",
                  "sampling_temperature", "num_translations_per_input",
                  "infer_mode"]

def add_arguments(parser):
  """Build ArgumentParser."""
  parser.register("type", "bool", lambda v: v.lower() == "true")

  parser.add_argument("--hparams_path", type=str, default=None,
                      help=("Path to standard hparams json file that overrides"
                            "hparams values from FLAGS."))
  parser.add_argument("--out_dir", type=str, default=None,
                      help="Store log/model files.")
  
  # Inference
  parser.add_argument("--ckpt", type=str, default="",
                      help="Checkpoint file to load a model for inference.")
  
  # Advanced inference arguments
  parser.add_argument("--infer_mode", type=str, default="greedy",
                      choices=["greedy", "sample", "beam_search"],
                      help="Which type of decoder to use during inference.")
  parser.add_argument("--beam_width", type=int, default=0,
                      help=("""\
      beam width when using beam search decoder. If 0 (default), use standard
      decoder with greedy helper.\
      """))
  parser.add_argument("--length_penalty_weight", type=float, default=0.0,
                      help="Length penalty for beam search.")
  parser.add_argument("--coverage_penalty_weight", type=float, default=0.0,
                      help="Coverage penalty for beam search.")
  parser.add_argument("--sampling_temperature", type=float,
                      default=0.0,
                      help=("""\
      Softmax sampling temperature for inference decoding, 0.0 means greedy
      decoding. This option is ignored when using beam search.\
      """))
  parser.add_argument("--num_translations_per_input", type=int, default=1,
                      help=("""\
      Number of translations generated for each sentence. This is only used for
      inference.\
      """))

def _get_user_input():
  """ Get user's input, which will be transformed into encoder input later """
  print("> ", end="")
  sys.stdout.flush()
  return sys.stdin.readline()

def run_chatbot(flags):

  # Load hparams.
  assert flags.hparams_path, "Need to provide a directory having a parameters"
  hparams = utils.load_hparams(flags.hparams_path)
  hparams.add_hparam("task", "joint")
  # Print HParams
  utils.print_hparams(hparams)

  ## Decode
  ckpt = flags.ckpt
  if not ckpt:
    # Model output directory
    out_dir = flags.out_dir
    assert out_dir, "Need to provide a directory having a trained model"
    ckpt = tf.train.latest_checkpoint(out_dir)

  model_creator = inference.get_model_creator(hparams)
  infer_model = model_helper.create_infer_model(model_creator, hparams, None)
  sess, loaded_infer_model = \
    inference.start_sess_and_load_model(infer_model, ckpt)

  # Begin chatbot for airline company client service
  print("Welcome to Alma airline customer service!!")
  print("I can only understand up to 50 words... Yes, I'm still learning :)\n")

  while True:
    user_input = _get_user_input()
    if len(user_input) > 0 and user_input[-1] == '\n':
      user_input = user_input[:-1]
    if user_input == '':
      break

    sess.run(
          infer_model.iterator.initializer,
          feed_dict={
              infer_model.src_placeholder: [user_input],
              infer_model.batch_size_placeholder: 1
          })  

    if hparams.task == "joint":
      outputs, intent_pred, src_seq_length, attention_summary = \
          loaded_infer_model.decode(sess)
    elif hparams.task == "intent":
      intent_pred, attention_summary = loaded_infer_model.decode(sess)

    if hparams.infer_mode == "beam_search":
      # get the top translation.
      outputs = outputs[0]

    if hparams.task == "joint":
      translation = nmt_utils.get_translation(
          outputs,
          src_seq_length,
          sent_id=0,
          tgt_eos=hparams.eos,
          subword_option=hparams.subword_option)  
      utils.print_out(b"   intent (pred): %s" % intent_pred[0])
      utils.print_out(b"     slot (pred): %s\n" % translation)
    elif hparams.task == "intent":
      utils.print_out(b"   intent (pred): %s\n" % intent_pred[0])
    
def main(unused_argv):
  run_chatbot(FLAGS)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
