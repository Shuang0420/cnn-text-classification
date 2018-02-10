# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import learn
import jieba
from .data_helpers import TextData


# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
# 1518167254
tf.flags.DEFINE_string("checkpoint_dir", "/home/shuang/sf/chatbot/wechat_yan/cnn_text_classification/cnnClassifier/save/1518167254/checkpoints", "Checkpoint directory from training run")
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



    def predict(self, input_, mode="rank"):
        input_ = self.process_input(input_)

        batches = self.predictData.batch_iter(input_, FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = self.sess.run(self.predictions, {self.input_x: x_test_batch, self.dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

        label_id = int(all_predictions[0])

        if mode=="rank":
            return self.predictData.id2label(label_id)
        return str(label_id)




    def process_input(self, input_):
        input_ = [" ".join(jieba.lcut(input_))]
        x = np.array(list(self.vocab_processor.transform(input_)))
        return x




def livePredict():
    model = PredictModel()
    while True:
        print(model.predict(input("> ")))


def eval():
    model = PredictModel()
    with open("./data/faq/faq_seg.txt") as fr:
        for line in fr:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            predict = model.predict(parts[0].replace(" ", ""))
            label = parts[1].replace(" ", "")
            if predict != label:
                print("query %s \t predict %s \t true %s" % (parts[0], predict, label))

# eval()
# livePredict()
