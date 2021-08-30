import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# import logging, os
# logging.disable(logging.WARNING)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import tensorflow as tf

from model.kws_model import FewShotKWS
from dataloader.dataloader import DataLoader
from trainer.kws_runners import KwsTrainer


class KwsTrain:
    def __init__(self):
        train_file = './data_txt/debug_list.txt'
        dev_file = './data_txt/debug_list.txt'
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
        train_loader = train_loader.shuffle(buffer_size=4000)
        self.train_loader = train_loader.padded_batch(4, padded_shapes=([600, 39, 1], []))

        val_l = self.data_loader.generator(train=False)
        val_loader = tf.data.Dataset.from_generator(lambda: val_l,
                                                      output_signature=(
                                                          tf.TensorSpec(shape=(None, None, 1), dtype=tf.float64),
                                                          tf.TensorSpec(shape=(), dtype=tf.int64)
                                                      ))
        # val_loader = val_loader.shuffle(buffer_size=4000)
        val_loader = val_loader.padded_batch(4, padded_shapes=([600, 39, 1], []))

        self.trainer = KwsTrainer(train_loader, val_loader, self.model)
        self.trainer.compile()


    def train(self):
        self.trainer._train_step()
        # self.trainer = KwsTrainer()


if __name__ == '__main__':
    train_file = './data_txt/train_list.txt'
    dev_file = './data_txt/val_list.txt'
    commands = './data_txt/commands.txt'
    with open(commands, 'r') as f:
        lines = f.readlines()
    with open(train_file, 'r') as f:
        train_list = f.readlines()
    with open(dev_file, 'r') as f:
        dev_list = f.readlines()
    num_label = len(lines)
    model = FewShotKWS(num_label)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./model/kws_test/checkpoints/model.{epoch:03d}',
        mode="max",
        monitor="val_accuracy",
        save_best_only=True,
    )


    data_loader = DataLoader(train_file, dev_file, commands)

    # train_l = data_loader.generator(train=True)
    train_loader = tf.data.Dataset.from_generator(data_loader.generator,
                                                  output_signature=(
                                                      tf.TensorSpec(shape=(None, None, 1), dtype=tf.float64),
                                                      tf.TensorSpec(shape=(), dtype=tf.int64)
                                                  ),
                                                  args=[True])
    train_loader = train_loader.shuffle(buffer_size=4000)
    train_loader = train_loader.padded_batch(8, padded_shapes=([600, 39, 1], []), drop_remainder=True)

    # val_l = data_loader.generator(train=False)
    val_loader = tf.data.Dataset.from_generator(data_loader.generator,
                                                output_signature=(
                                                    tf.TensorSpec(shape=(None, None, 1), dtype=tf.float64),
                                                    tf.TensorSpec(shape=(), dtype=tf.int64)
                                                ),
                                                args=[False])
    # val_loader = val_loader.shuffle(buffer_size=4000)
    val_loader = val_loader.padded_batch(4, padded_shapes=([600, 39, 1], []), drop_remainder=True)


    history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=40,
        callbacks=[model_checkpoint_callback],
    )

    tf.keras.models.save_model(
        model, './model/kws_test/checkpoints/model_last',
        options=tf.saved_model.SaveOptions(experimental_custom_gradients=True)
    )

    #
    # history_idx = 0
    # while os.path.isfile(f"./history_keras_{history_idx}.pkl"):
    #     history_idx += 1
    # with open(f"./history_keras_{history_idx}.pkl", "wb") as fh:
    #     pickle.dump(history.history, fh)

    print('asdfjasldkfjaskldfjaklsjdfklznxcvkljnaselkfnasjlhnvjklrngjkahnjkldsfnjkasb')






