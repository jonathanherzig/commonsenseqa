"""Run BERT on CommonsenseQA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os

import numpy as np
import tensorflow as tf

import modeling
import optimization
import tokenization


flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
  "split", None,
  "The split you'd like to run on, either 'qtoken' or 'rand'.")


## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  """A single multiple choice question."""

  def __init__(
      self,
      qid,
      question,
      answers,
      label):
    """Construct an instance."""
    self.qid = qid
    self.question = question
    self.answers = answers
    self.label = label


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_json(cls, input_file):
    """Reads a JSON file."""
    with tf.gfile.Open(input_file, "r") as f:
      return json.load(f)

  @classmethod
  def _read_jsonl(cls, input_file):
    """Reads a JSON Lines file."""
    with tf.gfile.Open(input_file, "r") as f:
      return [json.loads(ln) for ln in f]


class CommonsenseQAProcessor(DataProcessor):
  """Processor for the CommonsenseQA data set."""

  SPLITS = ['qtoken', 'rand']
  LABELS = ['A', 'B', 'C', 'D', 'E']

  TRAIN_FILE_NAME = 'train_{split}_split.jsonl'
  DEV_FILE_NAME = 'dev_{split}_split.jsonl'
  TEST_FILE_NAME = 'test_{split}_split_no_answers.jsonl'

  def __init__(self, split):
    if split not in self.SPLITS:
      raise ValueError(
        'split must be one of {", ".join(self.SPLITS)}.')

    self.split = split

  def get_train_examples(self, data_dir):
    train_file_name = self.TRAIN_FILE_NAME.format(split=self.split)

    return self._create_examples(
      self._read_jsonl(os.path.join(data_dir, train_file_name)),
      'train')

  def get_dev_examples(self, data_dir):
    dev_file_name = self.DEV_FILE_NAME.format(split=self.split)

    return self._create_examples(
      self._read_jsonl(os.path.join(data_dir, dev_file_name)),
      'dev')

  def get_test_examples(self, data_dir):
    test_file_name = self.TEST_FILE_NAME.format(split=self.split)

    return self._create_examples(
      self._read_jsonl(os.path.join(data_dir, test_file_name)),
      'test')

  def get_labels(self):
    return [0, 1, 2, 3, 4]

  def _create_examples(self, lines, set_type):
    examples = []
    for line in lines:
      qid = line['id']
      question = tokenization.convert_to_unicode(line['question']['stem'])
      answers = np.array([
        tokenization.convert_to_unicode(choice['text'])
        for choice in sorted(
            line['question']['choices'],
            key=lambda c: c['label'])
      ])
      # the test set has no answer key so use 'A' as a dummy label
      label = self.LABELS.index(line.get('answerKey', 'A'))

      examples.append(
        InputExample(
          qid=qid,
          question=question,
          answers=answers,
          label=label))

    return examples


def example_to_token_ids_segment_ids_label_ids(
    ex_index,
    example,
    max_seq_length,
    tokenizer):
  """Converts an ``InputExample`` to token ids and segment ids."""
  if ex_index < 5:
    tf.logging.info(f"*** Example {ex_index} ***")
    tf.logging.info("qid: %s" % (example.qid))

  question_tokens = tokenizer.tokenize(example.question)
  answers_tokens = map(tokenizer.tokenize, example.answers)

  token_ids = []
  segment_ids = []
  for choice_idx, answer_tokens in enumerate(answers_tokens):
    truncated_question_tokens = question_tokens[
      :max((max_seq_length - 3)//2, max_seq_length - (len(answer_tokens) + 3))]
    truncated_answer_tokens = answer_tokens[
      :max((max_seq_length - 3)//2, max_seq_length - (len(question_tokens) + 3))]

    choice_tokens = []
    choice_segment_ids = []
    choice_tokens.append("[CLS]")
    choice_segment_ids.append(0)
    for question_token in truncated_question_tokens:
      choice_tokens.append(question_token)
      choice_segment_ids.append(0)
    choice_tokens.append("[SEP]")
    choice_segment_ids.append(0)
    for answer_token in truncated_answer_tokens:
      choice_tokens.append(answer_token)
      choice_segment_ids.append(1)
    choice_tokens.append("[SEP]")
    choice_segment_ids.append(1)

    choice_token_ids = tokenizer.convert_tokens_to_ids(choice_tokens)

    token_ids.append(choice_token_ids)
    segment_ids.append(choice_segment_ids)

    if ex_index < 5:
      tf.logging.info("choice %s" % choice_idx)
      tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(t) for t in choice_tokens]))
      tf.logging.info("token ids: %s" % " ".join(
        [str(x) for x in choice_token_ids]))
      tf.logging.info("segment ids: %s" % " ".join(
        [str(x) for x in choice_segment_ids]))

  label_ids = [example.label]

  if ex_index < 5:
    tf.logging.info("label: %s (id = %d)" % (example.label, label_ids[0]))

  return token_ids, segment_ids, label_ids


def file_based_convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    output_file
):
  """Convert a set of ``InputExamples`` to a TFRecord file."""

  # encode examples into token_ids and segment_ids
  token_ids_segment_ids_label_ids = [
    example_to_token_ids_segment_ids_label_ids(
      ex_index,
      example,
      max_seq_length,
      tokenizer)
    for ex_index, example in enumerate(examples)
  ]

  # compute the maximum sequence length for any of the inputs
  seq_length = max([
    max([len(choice_token_ids) for choice_token_ids in token_ids])
    for token_ids, _, _ in token_ids_segment_ids_label_ids
  ])

  # encode the inputs into fixed-length vectors
  writer = tf.python_io.TFRecordWriter(output_file)

  for idx, (token_ids, segment_ids, label_ids) in enumerate(
      token_ids_segment_ids_label_ids
  ):
    if idx % 10000 == 0:
      tf.logging.info("Writing %d of %d" % (
        idx,
        len(token_ids_segment_ids_label_ids)))

    features = collections.OrderedDict()
    for i, (choice_token_ids, choice_segment_ids) in enumerate(
        zip(token_ids, segment_ids)):
      input_ids = np.zeros(max_seq_length)
      input_ids[:len(choice_token_ids)] = np.array(choice_token_ids)

      input_mask = np.zeros(max_seq_length)
      input_mask[:len(choice_token_ids)] = 1

      segment_ids = np.zeros(max_seq_length)
      segment_ids[:len(choice_segment_ids)] = np.array(choice_segment_ids)

      features[f'input_ids{i}'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(input_ids.astype(np.int64))))
      features[f'input_mask{i}'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(input_mask.astype(np.int64))))
      features[f'segment_ids{i}'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(segment_ids.astype(np.int64))))

    features['label_ids'] = tf.train.Feature(
      int64_list=tf.train.Int64List(value=label_ids))

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())

  return seq_length


def file_based_input_fn_builder(
    input_file,
    seq_length,
    is_training,
    drop_remainder
):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids0": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask0": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids0": tf.FixedLenFeature([seq_length], tf.int64),
      "input_ids1": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask1": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids1": tf.FixedLenFeature([seq_length], tf.int64),
      "input_ids2": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask2": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids2": tf.FixedLenFeature([seq_length], tf.int64),
      "input_ids3": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask3": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids3": tf.FixedLenFeature([seq_length], tf.int64),
      "input_ids4": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask4": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids4": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def create_model(
    bert_config,
    is_training,
    input_ids0,
    input_mask0,
    segment_ids0,
    input_ids1,
    input_mask1,
    segment_ids1,
    input_ids2,
    input_mask2,
    segment_ids2,
    input_ids3,
    input_mask3,
    segment_ids3,
    input_ids4,
    input_mask4,
    segment_ids4,
    labels,
    num_labels,
    use_one_hot_embeddings
):
  """Creates a classification model."""
  input_ids = tf.stack(
    [
      input_ids0,
      input_ids1,
      input_ids2,
      input_ids3,
      input_ids4
    ],
    axis=1)
  input_mask = tf.stack(
    [
      input_mask0,
      input_mask1,
      input_mask2,
      input_mask3,
      input_mask4
    ],
    axis=1)
  segment_ids = tf.stack(
    [
      segment_ids0,
      segment_ids1,
      segment_ids2,
      segment_ids3,
      segment_ids4
    ],
    axis=1)

  _, num_choices, seq_length = input_ids.shape

  input_ids = tf.reshape(input_ids, (-1, seq_length))
  input_mask = tf.reshape(input_mask, (-1, seq_length))
  segment_ids = tf.reshape(segment_ids, (-1, seq_length))

  output_layer = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings
  ).get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  softmax_weights = tf.get_variable(
      "softmax_weights", [hidden_size, 1],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.reshape(
      tf.matmul(output_layer, softmax_weights),
      (-1, num_choices))

    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(
    bert_config,
    num_labels,
    init_checkpoint,
    learning_rate,
    num_train_steps,
    num_warmup_steps,
    use_tpu,
    use_one_hot_embeddings
):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids0 = features["input_ids0"]
    input_mask0 = features["input_mask0"]
    segment_ids0 = features["segment_ids0"]
    input_ids1 = features["input_ids1"]
    input_mask1 = features["input_mask1"]
    segment_ids1 = features["segment_ids1"]
    input_ids2 = features["input_ids2"]
    input_mask2 = features["input_mask2"]
    segment_ids2 = features["segment_ids2"]
    input_ids3 = features["input_ids3"]
    input_mask3 = features["input_mask3"]
    segment_ids3 = features["segment_ids3"]
    input_ids4 = features["input_ids4"]
    input_mask4 = features["input_mask4"]
    segment_ids4 = features["segment_ids4"]
    label_ids = features["label_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
      bert_config,
      is_training,
      input_ids0,
      input_mask0,
      segment_ids0,
      input_ids1,
      input_mask1,
      segment_ids1,
      input_ids2,
      input_mask2,
      segment_ids2,
      input_ids3,
      input_mask3,
      segment_ids3,
      input_ids4,
      input_mask4,
      segment_ids4,
      label_ids,
      num_labels,
      use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
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

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(label_ids, predictions)
        loss = tf.metrics.mean(per_example_loss)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=probabilities, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  processor = CommonsenseQAProcessor(split=FLAGS.split)

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

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
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    train_seq_length = file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    tf.logging.info("  Longest training sequence = %d", train_seq_length)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    eval_seq_length = file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", len(eval_examples))
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
    tf.logging.info("  Longest eval sequence = %d", eval_seq_length)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      # Eval will be slightly WRONG on the TPU because it will truncate
      # the last batch.
      eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    predict_seq_length = file_based_convert_examples_to_features(
      predict_examples, label_list,
      FLAGS.max_seq_length, tokenizer,
      predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_examples))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
    tf.logging.info("  Longest predict sequence = %d", predict_seq_length)

    if FLAGS.use_tpu:
      # Warning: According to tpu_estimator.py Prediction on TPU is an
      # experimental feature and hence not supported here
      raise ValueError("Prediction in TPU not supported")

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    test_predictions_file = os.path.join(
      FLAGS.output_dir,
      "test_results.csv")
    with tf.gfile.GFile(test_predictions_file, "w") as writer:
      tf.logging.info("***** Predict results *****")
      for example, prediction in zip(predict_examples, result):
        output_line = ",".join([
          str(example.qid),
          str(CommonsenseQAProcessor.LABELS[np.argmax(prediction)])
        ] + [
          str(class_probability)
          for class_probability in prediction
        ]) + "\n"
        writer.write(output_line)


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
