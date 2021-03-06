# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 2019

@author: Sunhwan Lee (shlee@us.ibm.com)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import itertools
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import data_utils
import multi_task_model

import subprocess
import stat

'''
python test_multi-task_rnn.py --data_dir data/ATIS/processed \
      --train_dir model_tmp\
      --max_sequence_length 50 \
      --task joint
'''

#tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
#tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.9,
#                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 16,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("word_embedding_size", 128, "word embedding size")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("in_vocab_size", 10000, "max vocab Size.")
tf.app.flags.DEFINE_integer("out_vocab_size", 10000, "max tag vocab Size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit)")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("max_training_steps", 30000,
                            "Max training steps.")
tf.app.flags.DEFINE_integer("max_test_data_size", 0,
                            "Max size of test set.")
tf.app.flags.DEFINE_boolean("use_attention", True,
                            "Use attention based RNN")
tf.app.flags.DEFINE_integer("max_sequence_length", 0,
                            "Max sequence length.")
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
#_buckets = [(3, 10), (10, 25)]

def _softmax(logits):
  exps = [np.exp(i) for i in logits]
  sum_of_exps = sum(exps)
  softmax = [j/sum_of_exps for j in exps]

  return softmax

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')

  #print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=95)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               fontsize=8,
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()

# metrics function using conlleval.pl
def conlleval(p, g, w, filename):
  '''
  INPUT:
  p :: predictions
  g :: groundtruth
  w :: corresponding words

  OUTPUT:
  filename :: name of the file where the predictions
  are written. it will be the input of conlleval.pl script
  for computing the performance in terms of precision
  recall and f1 score
  '''
  out = ''
  for sl, sp, sw in zip(g, p, w):
    out += 'BOS O O\n'
    for wl, wp, w in zip(sl, sp, sw):
        out += w + ' ' + wl + ' ' + wp + '\n'
    out += 'EOS O O\n\n'

  if filename is not None:
    f = open(filename, 'w')
    f.writelines(out[:-1]) # remove the ending \n on last line
    f.close()

  return get_perf(filename)

def get_perf(filename):
  ''' run conlleval.pl perl script to obtain
  precision/recall and F1 score '''
  _conlleval = os.path.dirname(os.path.realpath(__file__)) + '/conlleval.pl'
  os.chmod(_conlleval, stat.S_IRWXU)  # give the execute permissions

  proc = subprocess.Popen(["perl",
                          _conlleval],
                          stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE,
                          encoding='utf8')

  stdout, _ = proc.communicate(''.join(open(filename).readlines()))
  for line in stdout.split('\n'):
    if 'accuracy' in line:
      out = line.split()
      break

  precision = float(out[6][:-2])
  recall = float(out[8][:-2])
  f1score = float(out[10])

  return {'p': precision, 'r': recall, 'f1': f1score}


def read_data(source_path, target_path, label_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the word sequence.
    target_path: path to the file with token-ids for the tag sequence;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    label_path: path to the file with token-ids for the intent label
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target, label) tuple read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1];source, target, label are lists of token-ids
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      with tf.gfile.GFile(label_path, mode="r") as label_file:
        source = source_file.readline()
        target = target_file.readline()
        label = label_file.readline()
        counter = 0
        while source and target and label and (not max_size \
                                               or counter < max_size):
          counter += 1
          if counter % 100000 == 0:
            print("  reading data line %d" % counter)
            sys.stdout.flush()
          source_ids = [int(x) for x in source.split()]
          target_ids = [int(x) for x in target.split()]
          label_ids = [int(x) for x in label.split()]
#          target_ids.append(data_utils.EOS_ID)
          for bucket_id, (source_size, target_size) in enumerate(_buckets):
            if len(source_ids) < source_size and len(target_ids) < target_size:
              data_set[bucket_id].append([source_ids, target_ids, label_ids])
              break
          source = source_file.readline()
          target = target_file.readline()
          label = label_file.readline()
  return data_set # 3 outputs in each unit: source_ids, target_ids, label_ids 

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
    model_test = multi_task_model.MultiTaskModel(
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
          task=task)

  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt:
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model_train.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model_train, model_test
        
def test():
  print ('Applying Parameters:')
  for k,v in FLAGS.__dict__['__flags'].items():
    print ('%s: %s' % (k, str(v)))
  print("\nPreparing data in %s" % FLAGS.data_dir)
  vocab_path = ''
  tag_vocab_path = ''
  label_vocab_path = ''
  date_set = data_utils.prepare_multi_task_data(
    FLAGS.data_dir, FLAGS.in_vocab_size, FLAGS.out_vocab_size)
  in_seq_test, out_seq_test, label_test = date_set[2]
  vocab_path, tag_vocab_path, label_vocab_path = date_set[3]
     
  vocab, rev_vocab = data_utils.initialize_vocab(vocab_path)
  tag_vocab, rev_tag_vocab = data_utils.initialize_vocab(tag_vocab_path)
  label_vocab, rev_label_vocab = data_utils.initialize_vocab(label_vocab_path)

  with tf.Session() as sess:
    # Create model.
    print("\nCreating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    
    model, model_test = create_model(sess, 
                                     len(vocab), 
                                     len(tag_vocab), 
                                     len(label_vocab))
    print ("Created model with " + 
           "source_vocab_size=%d, target_vocab_size=%d, label_vocab_size=%d." \
           % (len(vocab), len(tag_vocab), len(label_vocab)))

    # Read data into buckets and compute their sizes.
    print ("\nReading test data")
    test_set = read_data(in_seq_test, out_seq_test, label_test)
        
    def run_valid_test(data_set, mode): # mode: Eval, Test
    # Run evals on development/test set and print the accuracy.
        word_list = list() 
        ref_tag_list = list() 
        hyp_tag_list = list()
        ref_label_list = list()
        hyp_label_list = list()
        correct_count = 0
        accuracy = 0.0
        tagging_eval_result = dict()
        for bucket_id in xrange(len(_buckets)):
          eval_loss = 0.0
          count = 0
          for i in xrange(len(data_set[bucket_id])):
            count += 1
            sample = model_test.get_one(data_set, bucket_id, i)
            encoder_inputs,tags,tag_weights,sequence_length,labels = sample
            tagging_logits = []
            class_logits = []
            if task['joint'] == 1:
              step_outputs = model_test.joint_step(sess, 
                                                   encoder_inputs, 
                                                   tags, 
                                                   tag_weights, 
                                                   labels,
                                                   sequence_length, 
                                                   bucket_id, 
                                                   True)
              _, step_loss, tagging_logits, class_logits = step_outputs
              class_prob = _softmax(class_logits[0])
            elif task['tagging'] == 1:
              step_outputs = model_test.tagging_step(sess, 
                                                     encoder_inputs, 
                                                     tags, 
                                                     tag_weights,
                                                     sequence_length, 
                                                     bucket_id, 
                                                     True)
              _, step_loss, tagging_logits = step_outputs
            elif task['intent'] == 1:
              step_outputs = model_test.classification_step(sess, 
                                                            encoder_inputs, 
                                                            labels,
                                                            sequence_length, 
                                                            bucket_id, 
                                                            True) 
              _, step_loss, class_logits = step_outputs
            eval_loss += step_loss / len(data_set[bucket_id])
            hyp_label = None
            if task['intent'] == 1:
              ref_label_list.append(rev_label_vocab[labels[0][0]])
              hyp_label = np.argmax(class_logits[0],0)
              hyp_label_list.append(rev_label_vocab[hyp_label])
              if labels[0] == hyp_label:
                correct_count += 1
            if task['tagging'] == 1:
              word_list.append([rev_vocab[x[0]] for x in \
                                encoder_inputs[:sequence_length[0]]])
              ref_tag = [x[0] for x in tags[:sequence_length[0]]]
              ref_tag_list.append([rev_tag_vocab[x[0]] for x in \
                                   tags[:sequence_length[0]]])
              hyp_tag = [np.argmax(x) for x in tagging_logits[:sequence_length[0]]]
              hyp_tag_list.append(
                      [rev_tag_vocab[np.argmax(x)] for x in \
                                     tagging_logits[:sequence_length[0]]])
            
            if labels[0] != hyp_label or ref_tag != hyp_tag:
              error_type = []
              if labels[0] != hyp_label:
                error_type.append("Intent misclassification")
              if ref_tag != hyp_tag:
                error_type.append("Slot error")
              print("\n" + ", ".join(error_type))
              print("(intent) input: (%s) %s" % 
                      (rev_label_vocab[labels[0][0]], 
                       " ".join([rev_vocab[x[0]] for x in \
                                 encoder_inputs[:sequence_length[0]]])))
              print("true slots: %s" % " ".join(ref_tag_list[-1]))
              print("pred slots: %s" % " ".join(hyp_tag_list[-1]))
              pred_labels = np.argsort(class_prob)[-3:]
              intent_preds = [rev_label_vocab[l] for l in pred_labels]
              print("Top 3 predicted intents:")
              for idx in reversed(pred_labels):
                print("%s (%.4f)" % (rev_label_vocab[idx], class_prob[idx]))
        
        accuracy = float(correct_count)*100/count
        if task['intent'] == 1:
          print("  %s accuracy: %.2f %d/%d" \
                % (mode, accuracy, correct_count, count))
          sys.stdout.flush()

        '''
        if task['tagging'] == 1:
          tagging_eval_result = conlleval(hyp_tag_list, 
                                          ref_tag_list, 
                                          word_list, 
                                          None)
          print("  %s f1-score: %.2f" % (mode, tagging_eval_result['f1']))
          sys.stdout.flush()
        return accuracy, tagging_eval_result
        '''
        return accuracy, ref_label_list, hyp_label_list
        
    # test, run test after each validation for development purpose.
    #test_accuracy, test_tagging_result = run_valid_test(test_set, 'Test')
    test_accuracy, ref_label_list, hyp_label_list = run_valid_test(test_set, 'Test')

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(ref_label_list, hyp_label_list, labels=rev_label_vocab)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(12, 10))
    plot_confusion_matrix(cnf_matrix, classes=rev_label_vocab,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure(figsize=(12, 10))
    plot_confusion_matrix(cnf_matrix, classes=rev_label_vocab, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
          
def main(_):
  test()

if __name__ == "__main__":
  tf.app.run()
