"""
Created on Nov 6th 2018

@author: Sunhwan Lee

Process a raw dataset to a format for seq2seq model
"""

import os, re

from tensorflow.python.platform import gfile

import config

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]

def naive_tokenizer(sentence):
  """Naive tokenizer: split the sentence by space into a list of tokens."""
  return sentence.split()  

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
  vocab = {}
  with gfile.GFile(data_path, mode="r") as f:
    counter = 0
    for line in f:
      counter += 1
      if counter % 100000 == 0:
        print("  processing line %d" % counter)
      tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
      for w in tokens:
        word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
        if word in vocab:
          vocab[word] += 1
        else:
          vocab[word] = 1
    vocab_list = config.START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    if len(vocab_list) > max_vocabulary_size:
      vocab_list = vocab_list[:max_vocabulary_size]
    with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
      for w in vocab_list:
        vocab_file.write(w + "\n")

def create_label_vocab(vocabulary_path, data_path):
  print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
  vocab = {}
  with gfile.GFile(data_path, mode="r") as f:
    counter = 0
    for line in f:
      counter += 1
      if counter % 100000 == 0:
        print("  processing line %d" % counter)
      label = line.strip()
      vocab[label] = 1
    label_list = config.START_VOCAB + sorted(vocab)
    with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
      for k in label_list:
        vocab_file.write(k + "\n")

def prepare_dataset(inputfile, mode):

  input_seq_list = []
  output_seq_list = []
  intent_list = []
  intent_cnt = {}
  with gfile.GFile(inputfile, mode="r") as data_file:
    for line in data_file:
      input_seq = line[line.find('BOS')+4:line.find('EOS')].strip()
      output_seq = line[line.find('EOS')+4:line.find('atis_')].strip()
      # make sure that slot starts with O
      assert output_seq.startswith('O'), print("output_seq: %s" % output_seq)
      # exclude the first slot lable, O
      output_seq = output_seq[output_seq.find('O')+1:].strip()
      # make sure that input_seq and output_seq has the same number of words
      assert len(input_seq.split(' ')) == len(output_seq.split(' ')), \
      print("input: %s (%d), output:%s (%d)" % (input_seq, len(input_seq.split(' ')), \
                                                output_seq, len(output_seq.split(' '))))
      intent = line[line.find('atis_'):].strip()
      
      input_seq_list.append(input_seq)
      output_seq_list.append(output_seq)
      intent_list.append(intent)

  output_path = config.PROCESSED_DATA_PATH
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  input_seq_file = os.path.join(output_path, mode + '.seq.in')
  output_seq_file = os.path.join(output_path, mode + '.seq.out')
  label_file = os.path.join(output_path, mode + '.label')

  with gfile.GFile(input_seq_file, mode="w") as file:
    file.write('\n'.join(input_seq_list))

  with gfile.GFile(output_seq_file, mode="w") as file:
    file.write('\n'.join(output_seq_list))

  with gfile.GFile(label_file, mode="w") as file:
    file.write('\n'.join(intent_list))

  # create vocabulary if mode is train
  if mode == "train":

    # Create vocabularies of the appropriate sizes.
    in_vocab_path = os.path.join(output_path, "vocab.seq.in")
    out_vocab_path = os.path.join(output_path, "vocab.seq.out")
    label_path = os.path.join(output_path, "vocab.label")

    create_vocabulary(in_vocab_path, 
                      input_seq_file, 
                      config.VOCAB_SIZE, 
                      tokenizer=naive_tokenizer)
    create_vocabulary(out_vocab_path, 
                      output_seq_file, 
                      config.VOCAB_SIZE, 
                      tokenizer=naive_tokenizer)
    create_label_vocab(label_path, label_file)

def prepare_raw_data():
  print('Preparing raw data into train / test / validation set ...')
  raw_train_data = os.path.join(config.DATA_PATH, 'wta.train.intent')
  raw_test_data  = os.path.join(config.DATA_PATH, 'wta.test.intent')
  raw_valid_data = os.path.join(config.DATA_PATH, 'wta.valid.intent')
  #raw_train_data = os.path.join(config.DATA_PATH, 'atis.train.intent.iob')
  #raw_test_data  = os.path.join(config.DATA_PATH, 'atis.test.intent.iob')
  #raw_valid_data = os.path.join(config.DATA_PATH, 'atis-2.dev.intent.iob')
  prepare_dataset(raw_train_data, 'train')
  prepare_dataset(raw_test_data, 'test')
  prepare_dataset(raw_valid_data, 'valid')

if __name__ == '__main__':
  prepare_raw_data()