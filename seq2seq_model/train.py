"""
Created on Feb 22nd 2019

@author: Sunhwan Lee

For training Joint Slot Filling and Intent Detection model models.
"""
import math
import os
import random
import time

import tensorflow as tf

import attention_model
import inference
import model as base_model
import rnn_model
import model_helper
import utils
import nmt_utils

utils.check_tensorflow_version()

__all__ = [
    "run_sample_decode", "run_external_eval",
    "run_avg_external_eval", "run_full_eval", "init_stats", "update_stats",
    "print_step_info", "process_stats", "train", "get_model_creator",
    "add_info_summaries", "get_best_results"
]


def run_sample_decode(infer_model, infer_sess, model_dir, hparams,
                      summary_writer, src_data, tgt_data, lbl_data):
  """Sample decode a random sentence from src_data."""
  with infer_model.graph.as_default():
    loaded_infer_model, global_step = model_helper.create_or_load_model(
        infer_model.model, model_dir, infer_sess, "infer")

  _sample_decode(loaded_infer_model, global_step, infer_sess, hparams,
                 infer_model.iterator, src_data, tgt_data, lbl_data,
                 infer_model.src_placeholder,
                 infer_model.batch_size_placeholder, summary_writer)

def run_external_eval(infer_model,
                      infer_sess,
                      model_dir,
                      hparams,
                      summary_writer,
                      save_best_dev=True,
                      use_test_set=True,
                      avg_ckpts=False,
                      dev_infer_iterator_feed_dict=None,
                      test_infer_iterator_feed_dict=None):
  """Compute external evaluation for both dev / test.

  Computes development and testing external evaluation (e.g. f1, bleu, rouge) for
  given model.

  Args:
    infer_model: Inference model for which to compute perplexities.
    infer_sess: Inference TensorFlow session.
    model_dir: Directory from which to load inference model from.
    hparams: Model hyper-parameters.
    summary_writer: Summary writer for logging metrics to TensorBoard.
    use_test_set: Computes testing external evaluation if true; does not
      otherwise. Note that the development external evaluation is always
      computed regardless of value of this parameter.
    dev_infer_iterator_feed_dict: Feed dictionary for a TensorFlow session.
      Can be used to pass in additional inputs necessary for running the
      development external evaluation.
    test_infer_iterator_feed_dict: Feed dictionary for a TensorFlow session.
      Can be used to pass in additional inputs necessary for running the
      testing external evaluation.
  Returns:
    Triple containing development scores, testing scores and the TensorFlow
    Variable for the global step number, in this order.
  """
  if dev_infer_iterator_feed_dict is None:
    dev_infer_iterator_feed_dict = {}
  if test_infer_iterator_feed_dict is None:
    test_infer_iterator_feed_dict = {}
  with infer_model.graph.as_default():
    loaded_infer_model, global_step = model_helper.create_or_load_model(
        infer_model.model, model_dir, infer_sess, "infer")
  dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
  dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
  dev_lbl_file = "%s.%s" % (hparams.dev_prefix, hparams.lbl)
  dev_infer_iterator_feed_dict[
      infer_model.src_placeholder] = inference.load_data(dev_src_file)
  dev_infer_iterator_feed_dict[
      infer_model.batch_size_placeholder] = hparams.infer_batch_size
  dev_scores = _external_eval(
      loaded_infer_model,
      global_step,
      infer_sess,
      hparams,
      infer_model.iterator,
      dev_infer_iterator_feed_dict,
      dev_tgt_file,
      dev_lbl_file,
      "dev",
      summary_writer,
      save_on_best=save_best_dev)

  test_scores = None
  if use_test_set and hparams.test_prefix:
    test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
    test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
    test_lbl_file = "%s.%s" % (hparams.test_prefix, hparams.lbl)
    test_infer_iterator_feed_dict[
        infer_model.src_placeholder] = inference.load_data(test_src_file)
    test_infer_iterator_feed_dict[
        infer_model.batch_size_placeholder] = hparams.infer_batch_size
    test_scores = _external_eval(
        loaded_infer_model,
        global_step,
        infer_sess,
        hparams,
        infer_model.iterator,
        test_infer_iterator_feed_dict,
        test_tgt_file,
        test_lbl_file,
        "test",
        summary_writer,
        save_on_best=False)
  return dev_scores, test_scores, global_step

def run_full_eval(model_dir,
                  infer_model,
                  infer_sess,
                  eval_model,
                  eval_sess,
                  hparams,
                  summary_writer,
                  sample_src_data,
                  sample_tgt_data,
                  sample_lbl_data):
  """Wrapper for running sample_decode, internal_eval and external_eval.

  Args:
    model_dir: Directory from which to load models from.
    infer_model: Inference model for which to compute perplexities.
    infer_sess: Inference TensorFlow session.
    eval_model: Evaluation model for which to compute perplexities.
    eval_sess: Evaluation TensorFlow session.
    hparams: Model hyper-parameters.
    summary_writer: Summary writer for logging metrics to TensorBoard.
    sample_src_data: sample of source data for sample decoding.
    sample_tgt_data: sample of target data for sample decoding.
    sample_tgt_data: sample of label (intent) data for sample decoding.
  Returns:
    Triple containing results summary, global step Tensorflow Variable and
    metrics in this order.
  """
  run_sample_decode(infer_model, infer_sess, model_dir, hparams, summary_writer,
                    sample_src_data, sample_tgt_data, sample_lbl_data)
  
  dev_scores, test_scores, global_step = run_external_eval(
      infer_model,
      infer_sess,
      model_dir,
      hparams,
      summary_writer)

  metrics = {
      "dev_scores": dev_scores,
      "test_scores": test_scores,
  }

  result_summary = _format_results("dev", None, dev_scores, hparams.metrics)
  if hparams.test_prefix:
    result_summary += ", " + _format_results("test", None, test_scores,
                                             hparams.metrics)

  return result_summary, global_step, metrics

def init_stats():
  """Initialize statistics that we want to accumulate."""
  return {"step_time": 0.0, "slot_train_loss": 0.0, "intent_train_loss": 0.0,
          "predict_count": 0.0,  # word count on the target side
          "word_count": 0.0,  # word counts for both source and target
          "sequence_count": 0.0,  # number of training examples processed
          "grad_norm": 0.0}

def update_stats(hparams, stats, start_time, step_result):
  """Update stats: write summary and accumulate statistics."""
  _, output_tuple = step_result

  # Update statistics
  batch_size = output_tuple.batch_size
  stats["step_time"] += time.time() - start_time
  if hparams.task == "joint":
    stats["slot_train_loss"] += output_tuple.train_loss[0] * batch_size
    stats["intent_train_loss"] += output_tuple.train_loss[1] * batch_size
    stats["word_count"] += output_tuple.word_count
    stats["predict_count"] += output_tuple.predict_count
  elif hparams.task == "intent":
    stats["intent_train_loss"] += output_tuple.train_loss * batch_size
  stats["grad_norm"] += output_tuple.grad_norm
  stats["sequence_count"] += batch_size

  return (output_tuple.global_step, output_tuple.learning_rate,
          output_tuple.train_summary)

def print_step_info(prefix, global_step, info, result_summary, log_f):
  """Print all info at the current global step."""
  utils.print_out(
      "%sstep %d lr %g step-time %.2fs wps %.2fK gN %.2f %s, %s" %
      (prefix, global_step, info["learning_rate"], info["avg_step_time"],
       info["speed"], info["avg_grad_norm"], result_summary,
       time.ctime()),
      log_f)

def add_info_summaries(summary_writer, global_step, info):
  """Add stuffs in info to summaries."""
  excluded_list = ["learning_rate"]
  for key in info:
    if key not in excluded_list:
      utils.add_summary(summary_writer, global_step, key, info[key])

def process_stats(hparams, stats, info, global_step, steps_per_stats, log_f):
  """Update info and check for overflow."""
  # Per-step info
  info["avg_step_time"] = stats["step_time"] / steps_per_stats
  info["avg_grad_norm"] = stats["grad_norm"] / steps_per_stats
  info["avg_sequence_count"] = stats["sequence_count"] / steps_per_stats
  info["speed"] = stats["word_count"] / (1000 * stats["step_time"])

  # Check for overflow
  is_overflow = False
  if hparams.task == "joint":
    # Per-predict info
    info["train_ppl"] = (
        utils.safe_exp(stats["slot_train_loss"] / stats["predict_count"]))
  
    train_ppl = info["train_ppl"]
    if math.isnan(train_ppl) or math.isinf(train_ppl) or train_ppl > 1e20:
      utils.print_out("  step %d overflow, stop early" % global_step,
                      log_f)
      is_overflow = True

  return is_overflow

def before_train(loaded_train_model, train_model, train_sess, global_step,
                 hparams, log_f):
  """Misc tasks to do before training."""
  stats = init_stats()
  info = {"train_ppl": 0.0, "speed": 0.0,
          "avg_step_time": 0.0,
          "avg_grad_norm": 0.0,
          "avg_sequence_count": 0.0,
          "learning_rate": loaded_train_model.learning_rate.eval(
              session=train_sess)}
  start_train_time = time.time()
  utils.print_out("# Start step %d, lr %g, %s" %
                  (global_step, info["learning_rate"], time.ctime()), log_f)

  # Initialize all of the iterators
  skip_count = hparams.batch_size * hparams.epoch_step
  utils.print_out("# Init train iterator, skipping %d elements" % skip_count)
  train_sess.run(
      train_model.iterator.initializer,
      feed_dict={train_model.skip_count_placeholder: skip_count})

  return stats, info, start_train_time


def get_model_creator(hparams):
  """Get the right model class depending on configuration."""
  if hparams.attention and hparams.attention_architecture == "standard":
    model_creator = attention_model.AttentionModel
  elif not hparams.attention:
    if hparams.task == "joint":
      model_creator = base_model.Model
    elif hparams.task == "intent":
      model_creator = rnn_model.Model
  else:
    raise ValueError("Unknown attention architecture %s" %
                     hparams.attention_architecture)
  return model_creator

def train(hparams, scope=None, target_session=""):
  """Train a translation model."""
  log_device_placement = hparams.log_device_placement
  out_dir = hparams.out_dir
  num_train_steps = hparams.num_train_steps
  steps_per_stats = hparams.steps_per_stats
  steps_per_external_eval = hparams.steps_per_external_eval
  steps_per_eval = 10 * steps_per_stats
  
  if not steps_per_external_eval:
    steps_per_external_eval = 5 * steps_per_eval

  # Create model
  model_creator = get_model_creator(hparams)
  train_model = model_helper.create_train_model(model_creator, hparams, scope)
  eval_model = model_helper.create_eval_model(model_creator, hparams, scope)
  infer_model = model_helper.create_infer_model(model_creator, hparams, scope)

  # Preload data for sample decoding.
  dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
  dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
  dev_lbl_file = "%s.%s" % (hparams.dev_prefix, hparams.lbl)
  sample_src_data = inference.load_data(dev_src_file)
  sample_tgt_data = inference.load_data(dev_tgt_file)
  sample_lbl_data = inference.load_data(dev_lbl_file)

  summary_name = "train_log"
  model_dir = hparams.out_dir

  # Log and output files
  log_file = os.path.join(out_dir, "log_%d" % time.time())
  log_f = tf.gfile.GFile(log_file, mode="a")
  utils.print_out("\n# log_file=%s" % log_file, log_f)
  
  # TensorFlow model
  config_proto = utils.get_config_proto(
      log_device_placement=log_device_placement)
  train_sess = tf.Session(
      target=target_session, config=config_proto, graph=train_model.graph)
  eval_sess = tf.Session(
      target=target_session, config=config_proto, graph=eval_model.graph)
  infer_sess = tf.Session(
      target=target_session, config=config_proto, graph=infer_model.graph)

  with train_model.graph.as_default():
    loaded_train_model, global_step = model_helper.create_or_load_model(
        train_model.model, model_dir, train_sess, "train")

  # Summary writer
  summary_writer = tf.summary.FileWriter(
      os.path.join(out_dir, summary_name), train_model.graph)

  # First evaluation
  run_full_eval(
      model_dir, infer_model, infer_sess,
      eval_model, eval_sess, hparams,
      summary_writer, sample_src_data,
      sample_tgt_data, sample_lbl_data)
  
  last_stats_step = global_step
  last_eval_step = global_step
  last_external_eval_step = global_step
  
  # This is the training loop.
  stats, info, start_train_time = before_train(
      loaded_train_model, train_model, train_sess, global_step, hparams, log_f)
  while global_step < num_train_steps:
    ### Run a step ###
    start_time = time.time()
    try:
      step_result = loaded_train_model.train(train_sess)
      hparams.epoch_step += 1
    except tf.errors.OutOfRangeError:
      # Finished going through the training dataset.  Go to next epoch.
      hparams.epoch_step = 0
      utils.print_out(
          "# Finished an epoch, step %d." % global_step)
      #run_sample_decode(infer_model, infer_sess, model_dir, hparams,
      #                  summary_writer, sample_src_data, sample_tgt_data)
      #run_external_eval(infer_model, infer_sess, model_dir, hparams,
      #                  summary_writer)

      train_sess.run(
          train_model.iterator.initializer,
          feed_dict={train_model.skip_count_placeholder: 0})
      continue

    # Process step_result, accumulate stats, and write summary
    global_step, info["learning_rate"], step_summary = update_stats(
        hparams, stats, start_time, step_result)
    summary_writer.add_summary(step_summary, global_step)
    
    # Once in a while, we print statistics.
    if global_step - last_stats_step >= steps_per_stats:
      last_stats_step = global_step
      is_overflow = process_stats(
          hparams, stats, info, global_step, steps_per_stats, log_f)
      print_step_info("  ", global_step, info, get_best_results(hparams),
                      log_f)
      if is_overflow:
        break

      # Reset statistics
      stats = init_stats()

    if global_step - last_eval_step >= steps_per_eval:
      last_eval_step = global_step
      utils.print_out("# Save eval, global step %d" % global_step)
      add_info_summaries(summary_writer, global_step, info)

      # Save checkpoint
      loaded_train_model.saver.save(
          train_sess,
          os.path.join(out_dir, "translate.ckpt"),
          global_step=global_step)

      # Evaluate on dev/test
      run_sample_decode(infer_model, infer_sess,
                        model_dir, hparams, summary_writer, sample_src_data,
                        sample_tgt_data, sample_lbl_data)
      run_external_eval(
          infer_model, infer_sess, model_dir,
          hparams, summary_writer)

  # Done training
  loaded_train_model.saver.save(
      train_sess,
      os.path.join(out_dir, "translate.ckpt"),
      global_step=global_step)

  (result_summary, _, final_eval_metrics) = (
      run_full_eval(
          model_dir, infer_model, infer_sess, eval_model, eval_sess, hparams,
          summary_writer, sample_src_data, sample_tgt_data, sample_lbl_data))
  print_step_info("# Final, ", global_step, info, result_summary, log_f)
  utils.print_time("# Done training!", start_train_time)

  summary_writer.close()

  utils.print_out("# Start evaluating saved best models.")
  for metric in hparams.metrics:
    best_model_dir = getattr(hparams, "best_" + metric + "_dir")
    summary_writer = tf.summary.FileWriter(
        os.path.join(best_model_dir, summary_name), infer_model.graph)
    result_summary, best_global_step, _ = run_full_eval(
        best_model_dir, infer_model, infer_sess, eval_model, eval_sess, hparams,
        summary_writer, sample_src_data, sample_tgt_data, sample_lbl_data)
    print_step_info("# Best %s, " % metric, best_global_step, info,
                    result_summary, log_f)
    summary_writer.close()

  return final_eval_metrics, global_step

def _format_results(name, ppl, scores, metrics):
  """Format results."""
  result_str = ""
  if ppl:
    result_str = "%s ppl %.2f" % (name, ppl)
  if scores:
    for metric in metrics:
      if result_str:
        result_str += ", %s %s %.1f" % (name, metric, scores[metric])
      else:
        result_str = "%s %s %.1f" % (name, metric, scores[metric])
  
  return result_str


def get_best_results(hparams):
  """Summary of the current best results."""
  tokens = []
  for metric in hparams.metrics:
    tokens.append("%s %.2f" % (metric, getattr(hparams, "best_" + metric)))
  return ", ".join(tokens)


def _internal_eval(model, global_step, sess, iterator, iterator_feed_dict,
                   summary_writer, label):
  """Computing perplexity."""
  sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
  ppl = model_helper.compute_perplexity(model, sess, label)
  utils.add_summary(summary_writer, global_step, "%s_ppl" % label, ppl)
  return ppl


def _sample_decode(model, global_step, sess, hparams, iterator, src_data,
                   tgt_data, lbl_data, iterator_src_placeholder,
                   iterator_batch_size_placeholder, summary_writer):
  """Pick a sentence and decode."""
  decode_id = random.randint(0, len(src_data) - 1)
  utils.print_out("  # %d" % decode_id)

  iterator_feed_dict = {
      iterator_src_placeholder: [src_data[decode_id]],
      iterator_batch_size_placeholder: 1,
  }
  sess.run(iterator.initializer, feed_dict=iterator_feed_dict)

  if hparams.task == "joint":
    outputs, intent_pred, src_seq_length, attention_summary = model.decode(sess)
  elif hparams.task == "intent":
    intent_pred, attention_summary = model.decode(sess)
  
  if hparams.infer_mode == "beam_search":
    # get the top translation.
    outputs = outputs[0]

  utils.print_out("          intent: %s" % lbl_data[decode_id])
  utils.print_out("             src: %s" % src_data[decode_id])
  if hparams.task == "joint":
    translation = nmt_utils.get_translation(
        outputs,
        src_seq_length,
        sent_id=0,
        tgt_eos=hparams.eos,
        subword_option=hparams.subword_option)  
    utils.print_out("      slot (ref): %s" % tgt_data[decode_id])
    utils.print_out(b"   intent (pred): %s" % intent_pred[0])
    utils.print_out(b"     slot (pred): %s\n" % translation)
  elif hparams.task == "intent":
    utils.print_out(b"   intent (pred): %s" % intent_pred[0])

  # Summary
  if attention_summary is not None:
    summary_writer.add_summary(attention_summary, global_step)

def _external_eval(model, global_step, sess, hparams, iterator,
                   iterator_feed_dict, tgt_file, lbl_file, label, 
                   summary_writer, save_on_best):
  """External evaluation such as BLEU and ROUGE scores."""
  out_dir = hparams.out_dir
  decode = global_step > 0

  if decode:
    utils.print_out("# External evaluation, global step %d" % global_step)

  sess.run(iterator.initializer, feed_dict=iterator_feed_dict)

  slot_output = os.path.join(out_dir, "slot_output_%s" % label)
  intent_output = os.path.join(out_dir, "intent_output_%s" % label)
  scores = nmt_utils.decode_and_evaluate(
      label,
      model,
      sess,
      slot_output,
      intent_output,
      ref_file=tgt_file,
      ref_lbl_file=lbl_file,
      metrics=hparams.metrics,
      subword_option=hparams.subword_option,
      beam_width=hparams.beam_width,
      tgt_eos=hparams.eos,
      task=hparams.task,
      decode=decode,
      infer_mode=hparams.infer_mode)
  # Save on best metrics
  if decode:
    for metric in hparams.metrics:
      best_metric_label = "best_" + metric

      utils.add_summary(summary_writer, global_step, "%s_%s" % (label, metric),
                        scores[metric])
      # metric: larger is better
      if save_on_best and scores[metric] > getattr(hparams, best_metric_label):
        setattr(hparams, best_metric_label, scores[metric])
        model.saver.save(
            sess,
            os.path.join(
                getattr(hparams, best_metric_label + "_dir"), "translate.ckpt"),
            global_step=model.global_step)
    utils.save_hparams(out_dir, hparams)
  return scores
