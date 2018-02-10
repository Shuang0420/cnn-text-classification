# -*- coding: utf-8 -*-
import numpy as np
import os


class TextData:
    def __init__(self, standQf="./data/faq/standFAQ.txt"):
        self.standQ = None
        self.standQf = standQf


    def transform_label(self):
        """
        label_text => label_id
        :return:
        """
        standQ = open(self.standQf, 'r').readlines()
        standQ = [line.split("\t")[0] for line in standQ]
        self.standQ = standQ


    def id2label(self, label):
        """
        label_id => label_text
        :return: text
        """
        if not self.standQ:
            self.transform_label()
        return self.standQ[label]



    def transform_data(self, userQf, mode="rank"):
        """
        Transforms various data format into standard one:
            Data format in ranking mode:
            userQ \t self.standQ => userQ##label
            Data format in intent mode:
            __label__faq \t userQ => userQ##label
        :return:
        """

        if mode=="rank":
            self.transform_label()


        clean = userQf.replace(".txt", "") + ".clean"
        if os.path.exists(clean):
            return clean

        content = []
        with open(userQf, "r") as fr:
            for line in fr:
                parts = line.strip().split("\t")
                if len(parts) < 2: continue

                if mode=="rank":
                    label = self.standQ.index(parts[1].replace(" ", ""))
                    content.append("%s##%s\n" % (parts[0], label))
                else:
                    label = int(parts[0].replace("__label__", "")=="faq")
                    content.append("%s##%s\n" % (parts[1], label))

        fw = open(clean, "w")
        fw.writelines(content)
        return userQf



    def load_data_and_labels(self, userQf, mode="rank", nlabels=0):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Data format:
            seg_query##label
            人工 客服##368
        Returns split sentences and labels.
        """
        # Transforms data if userQf is not in regular data format. Otherwise just comment these lines
        if mode=="rank" and not userQf.endswith(".clean"):
            userQf = self.transform_data(userQf)
            if not nlabels: nlabels = len(self.standQ)
        else: # intent mode, default class label is 2
            userQf = self.transform_data(userQf, mode="intent")


        # Load data from files
        x_text, x_labels = [], []
        with open(userQf, "r") as fr:
            for line in fr:
                parts = line.strip().split("##")
                if len(parts) < 2: continue
                x_text.append(parts[0])
                x_labels.append(int(parts[1]))
        if not nlabels: nlabels = len(set(x_labels))
        y = []
        for label in x_labels:
            l = np.zeros(nlabels,int)
            l[label] = 1
            y.append(l)
        return [x_text, y]



    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]