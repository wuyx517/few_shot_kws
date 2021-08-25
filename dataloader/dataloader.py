import os

import numpy as np
import tensorflow as tf

from dataloader.data_wav_tools import read_wav_data, get_mfcc_feature


class DataLoader:
    def __init__(self, train_file_list, dev_file_list, commands):
        self.train_list = self.make_data_list(train_file_list)
        self.dev_list = self.make_data_list(dev_file_list)
        self.commands = np.array(self.make_data_list(commands))

    def make_data_list(self, file_list):
        if not isinstance(file_list, list):
            file_list = [file_list]
        data_list = []
        for file in file_list:
            with open(file, 'r') as f:
                data = f.readlines()
            data_list.extend(data)
        return data_list

    def generator(self, train=False):
        if train:
            data_list = self.train_list
        else:
            data_list = self.dev_list
        for data in data_list:
            wave_data, framerate = read_wav_data(data.strip())
            spec = get_mfcc_feature(wave_data)
            spec = tf.expand_dims(spec, -1)
            label = data.split('/')[-2]
            label_list = (label == self.commands)
            label_id = tf.argmax(label_list)
            yield spec, label_id


if __name__ == '__main__':
    data_l = DataLoader('../data_txt/train_list.txt', '../data_txt/val_list.txt', '../data_txt/commands.txt')
    train_l = data_l.generator(train=True)
    train_loader = tf.data.Dataset.from_generator(lambda: train_l,
                                                  output_signature=(
                                                      tf.TensorSpec(shape=(None, None, 1), dtype=tf.float64),
                                                      tf.TensorSpec(shape=(), dtype=tf.int64)
                                                  ))
    train_loader = train_loader.padded_batch(4)
    for i in train_loader:
        print(len(i))
        print(i[0].shape)
        break
