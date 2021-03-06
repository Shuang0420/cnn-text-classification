#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib import learn

from data_helpers import TextData

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("root_dir", "/home/shuang/sf/chatbot/wechat_yan/", "folder where to look for the models and data")
tf.flags.DEFINE_string("model_dir", "faq-dim-100-filter-s23n10-l2-0.1-1518245956", "Model for evaluation")
tf.flags.DEFINE_string("input_file", "faq/faq_seg.txt", "Data source for evaluation")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))

input_path = os.path.abspath(os.path.join(FLAGS.root_dir, "data", FLAGS.input_file))
checkpoint_dir = os.path.abspath(os.path.join(FLAGS.root_dir, "save", FLAGS.model_dir, "checkpoints"))

print("{}={}".format("INPUT_PATH", input_path))
print("{}={}".format("CHECKPOINT_DIR", checkpoint_dir))
print("")

# CHANGE THIS: Load data. Load your own data here
td = TextData()

# if data format is " 转 人工##0"
# fname = input_path
# elif data format is " 转 人工 \t 请 稍等"
fname = td.transform_data(input_path)
# elif data format is "__label__faq \t 转 人工"
# fname = td.transform_data(input_path, mode="intent")

x_raw, y_test = [], []
with open(fname,'r') as fr:
    for line in fr:
        parts = line.strip().split("##")
        if len(parts) < 2: continue
        x_raw.append(parts[0])
        y_test.append(int(parts[1]))


# Map data into vocabulary
vocab_path = os.path.join(checkpoint_dir, os.pardir, "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))
print(x_test)

print("\nEvaluating...\n")

evalData = TextData()

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = evalData.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    #print "Predictions: {} \t Actual: {}".format(all_predictions,y_test)
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
