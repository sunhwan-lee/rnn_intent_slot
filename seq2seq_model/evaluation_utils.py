# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility for evaluating various tasks, e.g., translation & summarization."""
import codecs
import os
import re
import subprocess

import tensorflow as tf

import bleu
import rouge


__all__ = ["evaluate"]


def evaluate(ref_file, trans_file, metric, subword_option=None):
  """Pick a metric and evaluate depending on task."""
  # BLEU scores for translation task
  if metric.lower() == "bleu":
    evaluation_score = _bleu(ref_file, trans_file,
                             subword_option=subword_option)
  # ROUGE scores for summarization tasks
  elif metric.lower() == "rouge":
    evaluation_score = _rouge(ref_file, trans_file,
                              subword_option=subword_option)
  elif metric.lower() == "accuracy":
    evaluation_score = _accuracy(ref_file, trans_file)
  elif metric.lower() == "f1":
    evaluation_score = _f1(ref_file, trans_file)
  elif metric.lower() == "word_accuracy":
    evaluation_score = _word_accuracy(ref_file, trans_file)
  else:
    raise ValueError("Unknown metric %s" % metric)

  return evaluation_score


def _clean(sentence, subword_option):
  """Clean and handle BPE or SPM outputs."""
  sentence = sentence.strip()

  # BPE
  if subword_option == "bpe":
    sentence = re.sub("@@ ", "", sentence)

  # SPM
  elif subword_option == "spm":
    sentence = u"".join(sentence.split()).replace(u"\u2581", u" ").lstrip()

  return sentence


# Follow //transconsole/localization/machine_translation/metrics/bleu_calc.py
def _bleu(ref_file, trans_file, subword_option=None):
  """Compute BLEU scores and handling BPE."""
  max_order = 4
  smooth = False

  ref_files = [ref_file]
  reference_text = []
  for reference_filename in ref_files:
    with codecs.getreader("utf-8")(
        tf.gfile.GFile(reference_filename, "rb")) as fh:
      reference_text.append(fh.readlines())

  per_segment_references = []
  for references in zip(*reference_text):
    reference_list = []
    for reference in references:
      reference = _clean(reference, subword_option)
      reference_list.append(reference.split(" "))
    per_segment_references.append(reference_list)

  translations = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
    for line in fh:
      line = _clean(line, subword_option=None)
      translations.append(line.split(" "))

  # bleu_score, precisions, bp, ratio, translation_length, reference_length
  bleu_score, _, _, _, _, _ = bleu.compute_bleu(
      per_segment_references, translations, max_order, smooth)
  return 100 * bleu_score


def _rouge(ref_file, summarization_file, subword_option=None):
  """Compute ROUGE scores and handling BPE."""

  references = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(ref_file, "rb")) as fh:
    for line in fh:
      references.append(_clean(line, subword_option))

  hypotheses = []
  with codecs.getreader("utf-8")(
      tf.gfile.GFile(summarization_file, "rb")) as fh:
    for line in fh:
      hypotheses.append(_clean(line, subword_option=None))

  rouge_score_map = rouge.rouge(hypotheses, references)
  return 100 * rouge_score_map["rouge_l/f_score"]


def _accuracy(label_file, pred_file):
  """Compute accuracy, each line contains a label."""

  with codecs.getreader("utf-8")(tf.gfile.GFile(label_file, "rb")) as label_fh:
    with codecs.getreader("utf-8")(tf.gfile.GFile(pred_file, "rb")) as pred_fh:
      count = 0.0
      match = 0.0
      for label in label_fh:
        label = label.strip()
        pred = pred_fh.readline().strip()
        if label == pred:
          match += 1
        count += 1
  return 100 * match / count

def _word_accuracy(label_file, pred_file):
  """Compute accuracy on per word basis."""

  with codecs.getreader("utf-8")(tf.gfile.GFile(label_file, "rb")) as label_fh:
    with codecs.getreader("utf-8")(tf.gfile.GFile(pred_file, "rb")) as pred_fh:
      total_acc, total_count = 0., 0.
      for sentence in label_fh:
        labels = sentence.strip().split(" ")
        preds = pred_fh.readline().strip().split(" ")
        match = 0.0
        for pos in range(min(len(labels), len(preds))):
          label = labels[pos]
          pred = preds[pos]
          if label == pred:
            match += 1
        total_acc += 100 * match / max(len(labels), len(preds))
        total_count += 1
  return total_acc / total_count


def _moses_bleu(multi_bleu_script, tgt_test, trans_file, subword_option=None):
  """Compute BLEU scores using Moses multi-bleu.perl script."""

  # TODO(thangluong): perform rewrite using python
  # BPE
  if subword_option == "bpe":
    debpe_tgt_test = tgt_test + ".debpe"
    if not os.path.exists(debpe_tgt_test):
      # TODO(thangluong): not use shell=True, can be a security hazard
      subprocess.call("cp %s %s" % (tgt_test, debpe_tgt_test), shell=True)
      subprocess.call("sed s/@@ //g %s" % (debpe_tgt_test),
                      shell=True)
    tgt_test = debpe_tgt_test
  elif subword_option == "spm":
    despm_tgt_test = tgt_test + ".despm"
    if not os.path.exists(despm_tgt_test):
      subprocess.call("cp %s %s" % (tgt_test, despm_tgt_test))
      subprocess.call("sed s/ //g %s" % (despm_tgt_test))
      subprocess.call(u"sed s/^\u2581/g %s" % (despm_tgt_test))
      subprocess.call(u"sed s/\u2581/ /g %s" % (despm_tgt_test))
    tgt_test = despm_tgt_test
  cmd = "%s %s < %s" % (multi_bleu_script, tgt_test, trans_file)

  # subprocess
  # TODO(thangluong): not use shell=True, can be a security hazard
  bleu_output = subprocess.check_output(cmd, shell=True)

  # extract BLEU score
  m = re.search("BLEU = (.+?),", bleu_output)
  bleu_score = float(m.group(1))

  return bleu_score

def _f1(label_file, pred_file):
  """Compute accuracy on per word basis."""

  with codecs.getreader("utf-8")(tf.gfile.GFile(label_file, "rb")) as label_fh:
    with codecs.getreader("utf-8")(tf.gfile.GFile(pred_file, "rb")) as pred_fh:
      true_slots = []
      pred_slots = []
      for sentence in label_fh:
        true_slots.append(sentence.strip().split(" "))
        pred_slots.append(pred_fh.readline().strip().split(" "))

      f1_score, _, _ = _compute_f1_score(true_slots, pred_slots)

  return f1_score

# compute f1 score is modified from conlleval.pl
def __startOfChunk(prevTag, tag, prevTagType, tagType, chunkStart = False):
  if prevTag == 'B' and tag == 'B':
    chunkStart = True
  if prevTag == 'I' and tag == 'B':
    chunkStart = True
  if prevTag == 'O' and tag == 'B':
    chunkStart = True
  if prevTag == 'O' and tag == 'I':
    chunkStart = True

  if prevTag == 'E' and tag == 'E':
    chunkStart = True
  if prevTag == 'E' and tag == 'I':
    chunkStart = True
  if prevTag == 'O' and tag == 'E':
    chunkStart = True
  if prevTag == 'O' and tag == 'I':
    chunkStart = True

  if tag != 'O' and tag != '.' and prevTagType != tagType:
    chunkStart = True
  return chunkStart

def __endOfChunk(prevTag, tag, prevTagType, tagType, chunkEnd = False):
  if prevTag == 'B' and tag == 'B':
    chunkEnd = True
  if prevTag == 'B' and tag == 'O':
    chunkEnd = True
  if prevTag == 'I' and tag == 'B':
    chunkEnd = True
  if prevTag == 'I' and tag == 'O':
    chunkEnd = True

  if prevTag == 'E' and tag == 'E':
    chunkEnd = True
  if prevTag == 'E' and tag == 'I':
    chunkEnd = True
  if prevTag == 'E' and tag == 'O':
    chunkEnd = True
  if prevTag == 'I' and tag == 'O':
    chunkEnd = True

  if prevTag != 'O' and prevTag != '.' and prevTagType != tagType:
    chunkEnd = True
  return chunkEnd

def __splitTagType(tag):
  s = tag.split('-')
  if len(s) > 2 or len(s) == 0:
    raise ValueError('tag format wrong. it must be B-xxx.xxx')
  if len(s) == 1:
    tag = s[0]
    tagType = ""
  else:
    tag = s[0]
    tagType = s[1]
  return tag, tagType

def _compute_f1_score(correct_slots, pred_slots):
  correctChunk = {}
  correctChunkCnt = 0
  foundCorrect = {}
  foundCorrectCnt = 0
  foundPred = {}
  foundPredCnt = 0
  correctTags = 0
  tokenCount = 0
  for correct_slot, pred_slot in zip(correct_slots, pred_slots):
    inCorrect = False
    lastCorrectTag = 'O'
    lastCorrectType = ''
    lastPredTag = 'O'
    lastPredType = ''
    for c, p in zip(correct_slot, pred_slot):
      correctTag, correctType = __splitTagType(c)
      predTag, predType = __splitTagType(p)

      if inCorrect == True:
        if __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
           __endOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
           (lastCorrectType == lastPredType):
          inCorrect = False
          correctChunkCnt += 1
          if lastCorrectType in correctChunk:
            correctChunk[lastCorrectType] += 1
          else:
            correctChunk[lastCorrectType] = 1
        elif __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) != \
             __endOfChunk(lastPredTag, predTag, lastPredType, predType) or \
             (correctType != predType):
          inCorrect = False

      if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
         __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
         (correctType == predType):
        inCorrect = True

      if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True:
        foundCorrectCnt += 1
        if correctType in foundCorrect:
          foundCorrect[correctType] += 1
        else:
          foundCorrect[correctType] = 1

      if __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True:
        foundPredCnt += 1
        if predType in foundPred:
          foundPred[predType] += 1
        else:
          foundPred[predType] = 1

      if correctTag == predTag and correctType == predType:
        correctTags += 1

      tokenCount += 1

      lastCorrectTag = correctTag
      lastCorrectType = correctType
      lastPredTag = predTag
      lastPredType = predType

    if inCorrect == True:
      correctChunkCnt += 1
      if lastCorrectType in correctChunk:
        correctChunk[lastCorrectType] += 1
      else:
        correctChunk[lastCorrectType] = 1

  if foundPredCnt > 0:
    precision = 100*correctChunkCnt/foundPredCnt
  else:
    precision = 0

  if foundCorrectCnt > 0:
    recall = 100*correctChunkCnt/foundCorrectCnt
  else:
    recall = 0

  if (precision+recall) > 0:
    f1 = (2*precision*recall)/(precision+recall)
  else:
    f1 = 0

  return f1, precision, recall
