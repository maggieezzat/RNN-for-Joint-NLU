from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from data import *
from model import Model
from my_metrics import *
#from tensorflow.python import debug as tf_debug
import numpy as np


input_steps = 50
embedding_size = 64
hidden_size = 100
n_layers = 2
batch_size = 16
vocab_size = 871
slot_size = 122
intent_size = 22
epoch_num = 50


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):

  #def __init__(self, input_steps, embedding_size, hidden_size, vocab_size, slot_size,
  #intent_size, epoch_num, batch_size=16, n_layers=1):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params): 
    """The `model_fn` for the Estimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    encoder_inputs = features["encoder_inputs"]
    #The actual length of each sentence input without the padding
    encoder_inputs_actual_length = features["encoder_inputs_actual_length"]
    decoder_targets = features["decoder_targets"]
    intent_targets = features["intent_targets"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    #####################################
    #calculate the loss
    total_loss = 0
    #####################################

    tvars = tf.trainable_variables()

    initialized_variable_names = {}

    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)


    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op)

    #elif mode == tf.estimator.ModeKeys.EVAL:


    return output_spec

  return model_fn



def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  return example





def input_fn_builder(input_files,
                     max_seq_length=50,
                     is_training,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]
    input_steps = params["input_steps"]

    name_to_features = {
        "encoder_inputs":
            tf.FixedLenFeature([input_steps, batch_size], tf.int64),
        "encoder_inputs_actual_length":
            tf.FixedLenFeature([batch_size], tf.int64),
        "decoder_targets":
            tf.FixedLenFeature([batch_size, input_steps], tf.int64),
        "intent_targets":
            tf.FixedLenFeature([batch_size], tf.int64),
    }

    # For training, we want to shuffle the data.
    # For eval, we want no shuffling .
    d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
    d = d.repeat()
    if is_training:
      #d = d.shuffle(buffer_size=len(input_files))
      d = d.shuffle(buffer_size=100)
    

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d

  return input_fn


    







def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.gfile.MakeDirs(FLAGS.output_dir)

    tpu_cluster_resolver = None
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))



    train_examples = None
    num_train_steps = None
    num_warmup_steps = None



    if FLAGS.do_train:
        #######################################
        #Read training examples
        #######################################
        train_examples = read_squad_examples(
            input_file=FLAGS.train_file, is_training=True)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
        ###############################################################################

        # Pre-shuffle the input to avoid having to make a very large shuffle
        # buffer in in the `input_fn`.
        rng = random.Random(12345)
        rng.shuffle(train_examples)

    ###########################################
    #Adjust the model fn builder
    model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=False,
      use_one_hot_embeddings=False)
    ###############################################

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=False,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)
  