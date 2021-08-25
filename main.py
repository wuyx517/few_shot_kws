import tensorflow as tf

from model.kws_model import FewShotKWS
from dataloader.dataloader import DataLoader
from trainer.kws_runners import KwsTrainer

class KwsTrain:
    def __init__(self):
        train_file = './data_txt/train_list.txt'
        dev_file = './data_txt/val_list.txt'
        commands = './data_txt/commands.txt'
        with open(commands, 'r') as f:
            lines = f.readlines()
        num_label = len(lines)
        self.model = FewShotKWS(num_label)
        self.data_loader = DataLoader(train_file, dev_file, commands)
        train_l = self.data_loader.generator(train=True)
        train_loader = tf.data.Dataset.from_generator(lambda: train_l,
                                                      output_signature=(
                                                          tf.TensorSpec(shape=(None, None, 1), dtype=tf.float64),
                                                          tf.TensorSpec(shape=(), dtype=tf.int64)
                                                      ))
        train_loader = train_loader.padded_batch(4, padded_shapes=([600, 39, 1], []))
        self.trainer = KwsTrainer(train_loader, self.model)
        self.trainer.compile()


    def train(self):
        self.trainer._train_step()
        # self.trainer = KwsTrainer()

if __name__ == '__main__':
    kws = KwsTrain()
    kws.train()


