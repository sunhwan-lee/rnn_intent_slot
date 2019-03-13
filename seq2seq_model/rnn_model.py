"""
Created on Feb 22nd 2019

@author: Sunhwan Lee

Basic sequence-to-sequence model with dynamic RNN support.
"""
from __future__ import absolute_import
from __future__ import division

import abc
import collections
import numpy as np

import tensorflow as tf

import model_helper
import iterator_utils
import utils
import vocab_utils

utils.check_tensorflow_version()

__all__ = ["BaseModel", "Model"]

class TrainOutputTuple(collections.namedtuple(
    "TrainOutputTuple", ("train_summary", "train_loss",
                         "global_step", "word_count", "batch_size", "grad_norm",
                         "learning_rate"))):
  """To allow for flexibily in returing different outputs."""
  pass


class EvalOutputTuple(collections.namedtuple(
    "EvalOutputTuple", ("eval_loss", "batch_size"))):
  """To allow for flexibily in returing different outputs."""
  pass


class InferOutputTuple(collections.namedtuple(
    "InferOutputTuple", ("infer_logits", "infer_summary", "sample_intent"))):
  """To allow for flexibily in returing different outputs."""
  pass


class BaseModel(object):
  """RNN base class.
  """

  def __init__(self,
               hparams,
               mode,
               iterator,
               source_vocab_table,
               target_vocab_table,
               label_vocab_table,
               reverse_target_vocab_table=None,
               reverse_target_intent_vocab_table=None,
               scope=None,
               extra_args=None):
    """Create the model.

    Args:
      hparams: Hyperparameter configurations.
      mode: TRAIN | EVAL | INFER
      iterator: Dataset Iterator that feeds data.
      source_vocab_table: Lookup table mapping source words to ids.
      target_vocab_table: Lookup table mapping target words to ids.
      label_vocab_table: Lookup table mapping label word to id.
      reverse_target_vocab_table: Lookup table mapping ids to target words. Only
        required in INFER mode. Defaults to None.
      scope: scope of the model.
      extra_args: model_helper.ExtraArgs, for passing customizable functions.

    """
    # Set params
    self._set_params_initializer(hparams, mode, iterator,
                                 source_vocab_table, target_vocab_table,
                                 label_vocab_table, scope, extra_args)

    # Train graph
    res = self.build_graph(hparams, scope=scope)
    self._set_train_or_infer(res, reverse_target_vocab_table, 
                             reverse_target_intent_vocab_table, hparams)
    
    # Saver
    self.saver = tf.train.Saver(
        tf.global_variables(), max_to_keep=hparams.num_keep_ckpts)

  def _set_params_initializer(self,
                              hparams,
                              mode,
                              iterator,
                              source_vocab_table,
                              target_vocab_table,
                              label_vocab_table,
                              scope,
                              extra_args=None):
    """Set various params for self and initialize."""
    assert isinstance(iterator, iterator_utils.BatchedInput)
    self.iterator = iterator
    self.mode = mode
    self.src_vocab_table = source_vocab_table
    self.tgt_vocab_table = target_vocab_table
    self.lbl_vocab_table = label_vocab_table

    self.src_vocab_size = hparams.src_vocab_size
    self.tgt_vocab_size = hparams.tgt_vocab_size
    self.lbl_vocab_size = hparams.lbl_vocab_size
    self.num_gpus = hparams.num_gpus
    self.time_major = hparams.time_major

    self.dtype = tf.float32
    self.num_sampled_softmax = hparams.num_sampled_softmax

    # extra_args: to make it flexible for adding external customizable code
    self.single_cell_fn = None
    if extra_args:
      self.single_cell_fn = extra_args.single_cell_fn

    # Set num units
    self.num_units = hparams.num_units

    # Set num layers
    self.num_encoder_layers = hparams.num_encoder_layers
    self.num_decoder_layers = hparams.num_decoder_layers
    assert self.num_encoder_layers
    assert self.num_decoder_layers

    # Batch size
    self.batch_size = tf.size(self.iterator.source_sequence_length)

    # Global step
    self.global_step = tf.Variable(0, trainable=False)

    # Initializer
    self.random_seed = hparams.random_seed
    initializer = model_helper.get_initializer(
        hparams.init_op, self.random_seed, hparams.init_weight)
    tf.get_variable_scope().set_initializer(initializer)

    # Embeddings
    self.encoder_emb_lookup_fn = tf.nn.embedding_lookup
    self.init_embeddings(hparams, scope)

  def _set_train_or_infer(self, res, 
                          reverse_target_vocab_table,
                          reverse_target_intent_vocab_table,
                          hparams):
    """Set up training and inference."""
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:

      self.train_loss = res[1]
      self.word_count = tf.reduce_sum(
          self.iterator.source_sequence_length) + tf.reduce_sum(
              self.iterator.target_sequence_length)
    elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
      self.eval_loss = res[1]      
    elif self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_logits, _, self.label_pred = res
      self.sample_intent = reverse_target_intent_vocab_table.lookup(
          tf.to_int64(self.label_pred))

    params = tf.trainable_variables()

    # Gradients and SGD update operation for training the model.
    # Arrange for the embedding vars to appear at the beginning.
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.learning_rate = tf.constant(hparams.learning_rate)
      # warm-up
      self.learning_rate = self._get_learning_rate_warmup(hparams)
      # decay
      self.learning_rate = self._get_learning_rate_decay(hparams)

      # Optimizer
      if hparams.optimizer == "sgd":
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      elif hparams.optimizer == "adam":
        opt = tf.train.AdamOptimizer(self.learning_rate)
      else:
        raise ValueError("Unknown optimizer type %s" % hparams.optimizer)

      # Gradients
      gradients = tf.gradients(
          self.train_loss,
          params,
          colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)

      clipped_grads, grad_norm_summary, grad_norm = model_helper.gradient_clip(
          gradients, max_gradient_norm=hparams.max_gradient_norm)
      self.grad_norm_summary = grad_norm_summary
      self.grad_norm = grad_norm

      self.update = opt.apply_gradients(
          zip(clipped_grads, params), global_step=self.global_step)

      # Summary
      self.train_summary = self._get_train_summary()
    elif self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_summary = self._get_infer_summary(hparams)

    # Print trainable variables
    utils.print_out("# Trainable variables")
    utils.print_out("Format: <name>, <shape>, <(soft) device placement>")
    for param in params:
      utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                        param.op.device))

  def _get_learning_rate_warmup(self, hparams):
    """Get learning rate warmup."""
    warmup_steps = hparams.warmup_steps
    warmup_scheme = hparams.warmup_scheme
    utils.print_out("  learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" %
                    (hparams.learning_rate, warmup_steps, warmup_scheme))

    # Apply inverse decay if global steps less than warmup steps.
    # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
    # When step < warmup_steps,
    #   learing_rate *= warmup_factor ** (warmup_steps - step)
    if warmup_scheme == "t2t":
      # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
      warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
      inv_decay = warmup_factor**(
          tf.to_float(warmup_steps - self.global_step))
    else:
      raise ValueError("Unknown warmup scheme %s" % warmup_scheme)

    return tf.cond(
        self.global_step < hparams.warmup_steps,
        lambda: inv_decay * self.learning_rate,
        lambda: self.learning_rate,
        name="learning_rate_warmup_cond")

  def _get_decay_info(self, hparams):
    """Return decay info based on decay_scheme."""
    if hparams.decay_scheme in ["luong5", "luong10", "luong234"]:
      decay_factor = 0.5
      if hparams.decay_scheme == "luong5":
        start_decay_step = int(hparams.num_train_steps / 2)
        decay_times = 5
      elif hparams.decay_scheme == "luong10":
        start_decay_step = int(hparams.num_train_steps / 2)
        decay_times = 10
      elif hparams.decay_scheme == "luong234":
        start_decay_step = int(hparams.num_train_steps * 2 / 3)
        decay_times = 4
      remain_steps = hparams.num_train_steps - start_decay_step
      decay_steps = int(remain_steps / decay_times)
    elif not hparams.decay_scheme:  # no decay
      start_decay_step = hparams.num_train_steps
      decay_steps = 0
      decay_factor = 1.0
    elif hparams.decay_scheme:
      raise ValueError("Unknown decay scheme %s" % hparams.decay_scheme)
    return start_decay_step, decay_steps, decay_factor

  def _get_learning_rate_decay(self, hparams):
    """Get learning rate decay."""
    start_decay_step, decay_steps, decay_factor = self._get_decay_info(hparams)
    utils.print_out("  decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
                    "decay_factor %g" % (hparams.decay_scheme,
                                         start_decay_step,
                                         decay_steps,
                                         decay_factor))

    return tf.cond(
        self.global_step < start_decay_step,
        lambda: self.learning_rate,
        lambda: tf.train.exponential_decay(
            self.learning_rate,
            (self.global_step - start_decay_step),
            decay_steps, decay_factor, staircase=True),
        name="learning_rate_decay_cond")

  def init_embeddings(self, hparams, scope):
    """Init embeddings."""
    self.embedding_encoder, self.embedding_decoder = (
        model_helper.create_emb_for_encoder_and_decoder(
            share_vocab=hparams.share_vocab,
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            src_embed_size=self.num_units,
            tgt_embed_size=self.num_units,
            src_vocab_file=hparams.src_vocab_file,
            tgt_vocab_file=hparams.tgt_vocab_file,
            src_embed_file=hparams.src_embed_file,
            tgt_embed_file=hparams.tgt_embed_file,
            scope=scope,))

  def _get_train_summary(self):
    """Get train summary."""
    train_summary = tf.summary.merge(
        [tf.summary.scalar("lr", self.learning_rate),
         tf.summary.scalar("intent_loss", self.train_loss)] +
        self.grad_norm_summary)
    return train_summary

  def train(self, sess):
    """Execute train graph."""
    assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
    output_tuple = TrainOutputTuple(train_summary=self.train_summary,
                                    train_loss=self.train_loss,
                                    global_step=self.global_step,
                                    word_count=self.word_count,
                                    batch_size=self.batch_size,
                                    grad_norm=self.grad_norm,
                                    learning_rate=self.learning_rate)
    return sess.run([self.update, output_tuple])

  def eval(self, sess):
    """Execute eval graph."""
    assert self.mode == tf.contrib.learn.ModeKeys.EVAL
    output_tuple = EvalOutputTuple(eval_loss=self.eval_loss,
                                   batch_size=self.batch_size)
    return sess.run(output_tuple)

  def build_graph(self, hparams, scope=None):
    """Subclass must implement this method.

    Creates a sequence-to-sequence model with dynamic RNN decoder API.
    Args:
      hparams: Hyperparameter configurations.
      scope: VariableScope for the created subgraph; default "dynamic_seq2seq".

    Returns:
      A tuple of the form (logits, loss_tuple, final_context_state, sample_id),
      where:
        logits: float32 Tensor [batch_size x num_decoder_symbols].
        loss: loss = the total loss / batch_size.    
    """
    utils.print_out("\n# Creating %s graph ..." % self.mode)

    with tf.variable_scope(scope or "rnn", dtype=self.dtype):
      # Encoder
      self.encoder_outputs, encoder_state = self._build_encoder(hparams)
      fw_state, bw_state = encoder_state
      print('encoder_outputs: ', self.encoder_outputs.shape)
      print('fw_state.h: ', fw_state.h.shape)
      print('bw_state.h: ', bw_state.h.shape)

      # Linear layer for classification of intent
      encoder_last_state = tf.concat([fw_state.h, bw_state.h], axis=1)
      print('encoder_last_state: ', encoder_last_state.shape)
      print()

      encoder_output_size = encoder_last_state.get_shape()[1].value
      print('encoder_output_size: ', encoder_output_size)
      w = tf.get_variable('w', [encoder_output_size, self.lbl_vocab_size], dtype=tf.float32)
      w_t = tf.transpose(w)
      v = tf.get_variable('v', [self.lbl_vocab_size], dtype=tf.float32)
      
      # apply the linear layer    
      label_logits = tf.nn.xw_plus_b(encoder_last_state, w, v) 
      label_pred = tf.argmax(label_logits, 1)
      print('label_scores: ', label_logits.shape)
      print()

      ## Loss
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        with tf.device(model_helper.get_device_str(self.num_encoder_layers - 1,
                                                   self.num_gpus)):
          loss = self._compute_loss(label_logits)
      else:
        loss = tf.constant(0.0)

      return label_logits, loss, label_pred

  @abc.abstractmethod
  def _build_encoder(self, hparams):
    """Subclass must implement this.

    Build and run an RNN encoder.

    Args:
      hparams: Hyperparameters configurations.

    Returns:
      A tuple of encoder_outputs and encoder_state.
    """
    pass

  def _build_encoder_cell(self, hparams, num_layers, base_gpu=0):
    """Build a multi-layer RNN cell that can be used by encoder."""

    return model_helper.create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=self.num_units,
        num_layers=num_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        num_gpus=hparams.num_gpus,
        mode=self.mode,
        base_gpu=base_gpu,
        single_cell_fn=self.single_cell_fn)

  def _get_infer_maximum_iterations(self, hparams, source_sequence_length):
    """Maximum decoding steps at inference time."""
    if hparams.tgt_max_len_infer:
      maximum_iterations = hparams.tgt_max_len_infer
      utils.print_out("  decoding maximum_iterations %d" % maximum_iterations)
    else:
      # TODO(thangluong): add decoding_length_factor flag
      decoding_length_factor = 2.0
      max_encoder_length = tf.reduce_max(source_sequence_length)
      maximum_iterations = tf.to_int32(tf.round(
          tf.to_float(max_encoder_length) * decoding_length_factor))
    return maximum_iterations

  def _softmax_cross_entropy_loss(
      self, logits, decoder_cell_outputs, labels):
    """Compute softmax loss or sampled softmax loss."""
    if self.num_sampled_softmax > 0:

      is_sequence = (decoder_cell_outputs.shape.ndims == 3)

      if is_sequence:
        labels = tf.reshape(labels, [-1, 1])
        inputs = tf.reshape(decoder_cell_outputs, [-1, self.num_units])

      crossent = tf.nn.sampled_softmax_loss(
          weights=tf.transpose(self.output_layer.kernel),
          biases=self.output_layer.bias or tf.zeros([self.tgt_vocab_size]),
          labels=labels,
          inputs=inputs,
          num_sampled=self.num_sampled_softmax,
          num_classes=self.tgt_vocab_size,
          partition_strategy="div",
          seed=self.random_seed)

      if is_sequence:
        if self.time_major:
          crossent = tf.reshape(crossent, [-1, self.batch_size])
        else:
          crossent = tf.reshape(crossent, [self.batch_size, -1])

    else:
      crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits)

    return crossent

  def _compute_loss(self, label_logits):
    """Compute optimization loss."""
    # Compute loss for intent
    target_label = self.iterator.target_label

    # The label distributions using softmax function
    intent_loss = tf.losses.sparse_softmax_cross_entropy(
          labels=target_label, logits=label_logits)
    intent_loss /= tf.to_float(self.batch_size)

    return intent_loss

  def _get_infer_summary(self, hparams):
    del hparams
    return tf.no_op()

  def infer(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.INFER
    output_tuple = InferOutputTuple(infer_logits=self.infer_logits,
                                    infer_summary=self.infer_summary,
                                    sample_intent=self.sample_intent)
    return sess.run(output_tuple)

  def decode(self, sess):
    """Decode a batch.

    Args:
      sess: tensorflow session to use.

    Returns:
      A tuple consiting of outputs, infer_summary.
        outputs: of size [batch_size, time]
    """
    output_tuple = self.infer(sess)
    sample_intent = output_tuple.sample_intent
    infer_summary = output_tuple.infer_summary
    
    # make sure outputs is of shape [batch_size, time] or [beam_width,
    # batch_size, time] when using beam search.
    if self.time_major:
      sample_intent = sample_intent.transpose()

    return sample_intent, infer_summary

  def build_encoder_states(self, include_embeddings=False):
    """Stack encoder states and return tensor [batch, length, layer, size]."""
    assert self.mode == tf.contrib.learn.ModeKeys.INFER
    if include_embeddings:
      stack_state_list = tf.stack(
          [self.encoder_emb_inp] + self.encoder_state_list, 2)
    else:
      stack_state_list = tf.stack(self.encoder_state_list, 2)

    # transform from [length, batch, ...] -> [batch, length, ...]
    if self.time_major:
      stack_state_list = tf.transpose(stack_state_list, [1, 0, 2, 3])

    return stack_state_list


class Model(BaseModel):
  """Sequence-to-sequence dynamic model.

  This class implements a multi-layer recurrent neural network as encoder,
  and a multi-layer recurrent neural network decoder.
  """
  def _build_encoder_from_sequence(self, hparams, sequence, sequence_length):
    """Build an encoder from a sequence.

    Args:
      hparams: hyperparameters.
      sequence: tensor with input sequence data.
      sequence_length: tensor with length of the input sequence.

    Returns:
      encoder_outputs: RNN encoder outputs.
      encoder_state: RNN encoder state.

    Raises:
      ValueError: if encoder_type is neither "uni" nor "bi".
    """
    num_layers = self.num_encoder_layers
    
    if self.time_major:
      sequence = tf.transpose(sequence)

    with tf.variable_scope("encoder") as scope:
      dtype = scope.dtype

      self.encoder_emb_inp = self.encoder_emb_lookup_fn(
          self.embedding_encoder, sequence)

      # Encoder_outputs: [max_time, batch_size, num_units]
      if hparams.encoder_type == "uni":
        utils.print_out("  num_layers = %d" % num_layers)
        cell = self._build_encoder_cell(hparams, num_layers)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            cell,
            self.encoder_emb_inp,
            dtype=dtype,
            sequence_length=sequence_length,
            time_major=self.time_major,
            swap_memory=True)
      elif hparams.encoder_type == "bi":
        num_bi_layers = int(num_layers / 2)
        utils.print_out("  num_bi_layers = %d" % num_bi_layers)

        encoder_outputs, bi_encoder_state = (
            self._build_bidirectional_rnn(
                inputs=self.encoder_emb_inp,
                sequence_length=sequence_length,
                dtype=dtype,
                hparams=hparams,
                num_bi_layers=num_bi_layers))

        if num_bi_layers == 1:
          encoder_state = bi_encoder_state
        else:
          # alternatively concat forward and backward states
          encoder_state = []
          for layer_id in range(num_bi_layers):
            encoder_state.append(bi_encoder_state[0][layer_id])  # forward
            encoder_state.append(bi_encoder_state[1][layer_id])  # backward
          encoder_state = tuple(encoder_state)
      else:
        raise ValueError("Unknown encoder_type %s" % hparams.encoder_type)

    # Use the top layer for now
    self.encoder_state_list = [encoder_outputs]

    return encoder_outputs, encoder_state

  def _build_encoder(self, hparams):
    """Build encoder from source."""
    utils.print_out("# Build a basic encoder")
    return self._build_encoder_from_sequence(
        hparams, self.iterator.source, self.iterator.source_sequence_length)

  def _build_bidirectional_rnn(self, inputs, sequence_length,
                               dtype, hparams,
                               num_bi_layers,
                               base_gpu=0):
    """Create and call biddirectional RNN cells.

    Args:
      base_gpu: The gpu device id to use for the first forward RNN layer. The
        i-th forward RNN layer will use `(base_gpu + i) % num_gpus` as its
        device id. The `base_gpu` for backward RNN cell is `(base_gpu +
        num_bi_layers)`.

    Returns:
      The concatenated bidirectional output and the bidirectional RNN cell"s
      state.
    """
    # Construct forward and backward cells
    fw_cell = self._build_encoder_cell(hparams,
                                       num_bi_layers,
                                       base_gpu=base_gpu)
    bw_cell = self._build_encoder_cell(hparams,
                                       num_bi_layers,
                                       base_gpu=(base_gpu + num_bi_layers))

    bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
        fw_cell,
        bw_cell,
        inputs,
        dtype=dtype,
        sequence_length=sequence_length,
        time_major=self.time_major,
        swap_memory=True)

    return tf.concat(bi_outputs, -1), bi_state

  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length, base_gpu=0):
    """Build an RNN cell that can be used by decoder."""
    # We only make use of encoder_outputs in attention-based models
    if hparams.attention:
      raise ValueError("BasicModel doesn't support attention.")

    cell = model_helper.create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=self.num_units,
        num_layers=self.num_decoder_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        num_gpus=self.num_gpus,
        mode=self.mode,
        single_cell_fn=self.single_cell_fn,
        base_gpu=base_gpu
    )

    # For beam search, we need to replicate encoder infos beam_width times
    if (self.mode == tf.contrib.learn.ModeKeys.INFER and
        hparams.infer_mode == "beam_search"):
      decoder_initial_state = tf.contrib.seq2seq.tile_batch(
          encoder_state, multiplier=hparams.beam_width)
    else:
      decoder_initial_state = encoder_state

    return cell, decoder_initial_state
