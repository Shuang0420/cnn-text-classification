# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import data_helpers
import requests
from tensorflow.contrib import learn
# from config import MAX_SENTENCE_SIZE
# from config import WORD_DICTIONARY
# from config import WORD_EMBEDDING
# from config import CHECKPOINT_DIR
import time
import jieba
from data_helpers import TextData


# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
# tf.flags.DEFINE_string("checkpoint_dir", CHECKPOINT_DIR, "Checkpoint directory from training run")
tf.flags.DEFINE_string("checkpoint_dir", "./save/1518077170/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_string("vector_type", 'non', "Use randomly initialized vector(rand)/pre-trained vector(pre)/static and pre-trained vector(static) (default: rand)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")



class PredictModel(object):
    def __init__(self):
        self.predictData = TextData()
        if FLAGS.vector_type is 'static':
            self.word_dict, self.word2vec = data_helpers.load_embedding(WORD_DICTIONARY, WORD_EMBEDDING)
        else:
            vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
            self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        self.graph = tf.Graph()


        # Get the placeholders from the graph by name
        with self.graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            self.sess = tf.Session(config=session_conf)

            # Load the saved meta graph and restore variables
            self.saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            self.saver.restore(self.sess, checkpoint_file)

            with self.sess.as_default():
                self.input_x = self.graph.get_operation_by_name("input_x").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                self.dropout_keep_prob = self.graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                self.predictions = self.graph.get_operation_by_name("output/predictions").outputs[0]



    def predict(self, input):
        batches = self.predictData.batch_iter(input, FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = self.sess.run(self.predictions, {self.input_x: x_test_batch, self.dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

        return int(all_predictions[0])




def process_input(input, model):
    # if not model.word_dict and not model.vocab_processor:
    #     raise Exception("No word dictionary could be found!")
    input = [" ".join(jieba.lcut(input))]
    # Map data into vocabulary
    # if model.word_dict:
    #     x, none = data_helpers.map_vocabulary([input], None, MAX_SENTENCE_SIZE, model.word_dict)
    # elif model.vocab_processor:
    x = np.array(list(model.vocab_processor.transform(input)))
    return x




def livePredict():
    td = TextData()
    model = PredictModel()
    while True:
        _input = input("> ")
        x = process_input(_input, model)
        label = model.predict(x)
        print(td.id2label(label))


livePredict()
