# -*- coding: utf-8 -*-
"""
Created on Nov 6th 2018

@author: Sunhwan Lee
"""

import os

from tensorflow.python.platform import gfile

import config

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

  output_path = config.PROCESSED_DATA_PATH + '/' + mode
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

def prepare_raw_data():
  print('Preparing raw data into train / test / validation set ...')
  raw_train_data = os.path.join(config.DATA_PATH, 'atis.train.w-intent.iob')
  raw_test_data  = os.path.join(config.DATA_PATH, 'atis.test.w-intent.iob')
  raw_valid_data = os.path.join(config.DATA_PATH, 'atis-2.dev.w-intent.iob')
  prepare_dataset(raw_train_data, 'train')
  prepare_dataset(raw_test_data, 'test')
  prepare_dataset(raw_valid_data, 'valid')

if __name__ == '__main__':
  prepare_raw_data()