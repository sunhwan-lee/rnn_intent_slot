# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 2019

@author: Sunhwan Lee (shlee@us.ibm.com)
"""

import os
import sys
from pprint import pprint

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import multi_task_model

tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 16,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("word_embedding_size", 128, "word embedding size")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_string("data_dir", "data/ATIS/processed", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "model_tmp", "Training directory.")
tf.app.flags.DEFINE_integer("in_vocab_size", 10000, "max vocab Size.")
tf.app.flags.DEFINE_integer("out_vocab_size", 10000, "max tag vocab Size.")
tf.app.flags.DEFINE_integer("max_sequence_length", 0,
                            "Max sequence length.")
tf.app.flags.DEFINE_boolean("use_attention", True,
                            "Use attention based RNN")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5,
                          "dropout keep cell input and output prob.")  
tf.app.flags.DEFINE_boolean("bidirectional_rnn", True,
                            "Use birectional RNN")
tf.app.flags.DEFINE_string("task", None, "Options: joint; intent; tagging")
FLAGS = tf.app.flags.FLAGS

if FLAGS.max_sequence_length == 0:
    print ('Please indicate max sequence length. Exit')
    exit()

if FLAGS.task is None:
    print ('Please indicate task to run.' + 
           'Available options: intent; tagging; joint')
    exit()

task = dict({'intent':0, 'tagging':0, 'joint':0})
if FLAGS.task == 'intent':
    task['intent'] = 1
elif FLAGS.task == 'tagging':
    task['tagging'] = 1
elif FLAGS.task == 'joint':
    task['intent'] = 1
    task['tagging'] = 1
    task['joint'] = 1

_buckets = [(FLAGS.max_sequence_length, FLAGS.max_sequence_length)]

def create_model(session, 
                 source_vocab_size, 
                 target_vocab_size, 
                 label_vocab_size):
  """Create model and initialize or load parameters in session."""
  with tf.variable_scope("model", reuse=None):
    model_train = multi_task_model.MultiTaskModel(
          source_vocab_size, 
          target_vocab_size, 
          label_vocab_size, 
          _buckets,
          FLAGS.word_embedding_size, 
          FLAGS.size, 
          FLAGS.num_layers, 
          FLAGS.max_gradient_norm, 
          FLAGS.batch_size,
          dropout_keep_prob=FLAGS.dropout_keep_prob, 
          use_lstm=True,
          forward_only=False, 
          use_attention=FLAGS.use_attention,
          bidirectional_rnn=FLAGS.bidirectional_rnn,
          task=task)
  with tf.variable_scope("model", reuse=True):
    model_pred = multi_task_model.MultiTaskModel(
          source_vocab_size, 
          target_vocab_size, 
          label_vocab_size, 
          _buckets,
          FLAGS.word_embedding_size, 
          FLAGS.size, 
          FLAGS.num_layers, 
          FLAGS.max_gradient_norm, 
          FLAGS.batch_size,
          dropout_keep_prob=FLAGS.dropout_keep_prob, 
          use_lstm=True,
          forward_only=True, 
          use_attention=FLAGS.use_attention,
          bidirectional_rnn=FLAGS.bidirectional_rnn,
          task=task,
          pred_only=True)

  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt:
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model_train.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())

  return model_train, model_pred

def _softmax(logits):
  exps = [np.exp(i) for i in logits]
  sum_of_exps = sum(exps)
  softmax = [j/sum_of_exps for j in exps]

  return softmax

def _get_user_input():
  """ Get user's input, which will be transformed into encoder input later """
  print("> ", end="")
  sys.stdout.flush()
  return sys.stdin.readline()

def _extract_entity(tagging, user_input_to_model):

  entity_dict = {}
  for idx, tag in enumerate(tagging):
    if tag.startswith("B-"):
      entity_name = tag[2:]
      entity_value = user_input_to_model[idx]
      if entity_name in entity_dict:
        entity_dict[entity_name].append(entity_value)
      else:
        entity_dict[entity_name] = [entity_value]
      current_entity_name = entity_name
    elif tag.startswith("I-"):
      entity_name = tag[2:]
      entity_value = user_input_to_model[idx]
      if entity_name == current_entity_name:
        entity_dict[entity_name][-1] += (" " + entity_value)
      else:
        print("Predicted tag %s must be followed by B-%s tag" % (tag, entity_name))
        entity_dict[entity_name].append(entity_value)
    else:
      current_entity_name = None

  return entity_dict

def chatbot():

  print ('Applying Parameters:')
  for k,v in FLAGS.__dict__['__flags'].items():
    print ('%s: %s' % (k, str(v)))
  print('\n')
  
  vocab_path = os.path.join(FLAGS.data_dir, "in_vocab_%d.txt" % FLAGS.in_vocab_size)
  tag_vocab_path = os.path.join(FLAGS.data_dir, "out_vocab_%d.txt" % FLAGS.out_vocab_size)
  label_vocab_path = os.path.join(FLAGS.data_dir, "label.txt")

  vocab, rev_vocab = data_utils.initialize_vocab(vocab_path)
  tag_vocab, rev_tag_vocab = data_utils.initialize_vocab(tag_vocab_path)
  label_vocab, rev_label_vocab = data_utils.initialize_vocab(label_vocab_path)

  with tf.Session() as sess:

    # Create model.    
    print("\nLoading a model...")
    _, model_pred = create_model(sess, 
                                 len(vocab), 
                                 len(tag_vocab), 
                                 len(label_vocab))
    print ("Created model with " + 
           "source_vocab_size=%d, target_vocab_size=%d, label_vocab_size=%d." \
           % (len(vocab), len(tag_vocab), len(label_vocab)))
    print("\n")

    # Begin chatbot for airline company client service
    print("Welcome to Alma airline customer service!!")
    print("I can only understand up to 50 words... Yes, I'm still learning :)\n")

    # Create vocabularies of the appropriate sizes.
    '''
    user_input = "i want to fly from baltimore to dallas round trip"
    ref_tag = "O O O O O B-fromloc.city_name O B-toloc.city_name B-round_trip I-round_trip"
    ref_class = "atis_flight"
    print("\noriginal input: %s" % user_input)
    print("tagging: ", list(zip(user_input.split(" "),ref_tag.split(" "))))
    print("intent: %s\n" % ref_class)
    '''
    
    while True:
      user_input = _get_user_input()
      if len(user_input) > 0 and user_input[-1] == '\n':
        user_input = user_input[:-1]
      if user_input == '':
        break

      _, user_input_to_model, input_token_ids = \
        data_utils.user_input_to_token_ids(user_input, vocab_path)
      print("input to the model: %s" % user_input_to_model)
      #print("tokenized input:", input_token_ids)

      if (len(input_token_ids) > FLAGS.max_sequence_length):
        print('Max length I can handle is:', FLAGS.max_sequence_length)
        user_input = _get_user_input()
        continue

      for bucket_id, (source_size, target_size) in enumerate(_buckets):
        if len(input_token_ids) < source_size:
          input_bucket_id = bucket_id
          break
      #print("input bucket id: %d\n" % input_bucket_id)

      sample = model_pred.get_batch_format(input_token_ids, input_bucket_id)
      encoder_inputs,sequence_length = sample
      #print("encoder_inputs: ", encoder_inputs)
      #print("sequence_length: ", sequence_length)

      if task['joint'] == 1:
        pred_outputs = model_pred.joint_pred(sess, 
                                             encoder_inputs, 
                                             sequence_length, 
                                             input_bucket_id)
        tagging_logits, class_logits = pred_outputs
        class_prob = _softmax(class_logits[0])
        #print(class_logits[0])
        #print(class_prob)
      else:
        sys.exit("Current only support joint model")

      #print("tagging_logits: ", tagging_logits)
      #print("class_logits: ", class_logits)

      if task['intent'] == 1:
        hyp_label = np.argmax(class_logits[0],0)
        intent_pred = rev_label_vocab[hyp_label]
        print("Predicted intents: %s" % (intent_pred))
        pred_labels = np.argsort(class_prob)[-3:]
        intent_preds = [rev_label_vocab[l] for l in pred_labels]
        print("Top 3 predicted intents:")
        for idx in reversed(pred_labels):
          print("%s (%.4f)" % (rev_label_vocab[idx], class_prob[idx]))

      if task['tagging'] == 1:
        tag_pred = [rev_tag_vocab[np.argmax(x)] for x in \
                      tagging_logits[:sequence_length[0]]]
        print("predicted tag:", tag_pred)
        entity_dict = _extract_entity(tag_pred, user_input_to_model.split(" "))
        print("extraged entity:")
        pprint(entity_dict)
        print("\n")

def main(_):

  chatbot()

if __name__ == "__main__":
  tf.app.run()
