import os
import tensorflow as tf
from model.kws_model import TransferLearn, FewShotKWS
from dataloader.dataloader import DataLoader

class TrainTrans:
    def __init__(self):
        model_dir = ''
        target_train_list = ''
        target_dev_list = ''

        base_model = tf.keras.models.load_model(model_dir)
        transfer_model = TransferLearn(base_model, 4)


if __name__ == '__main__':
    model_dir = './test_model'
    target_train_list = './data_txt/target_train.txt'
    target_dev_list = './data_txt/target_val.txt'
    target_commands = './data_txt/commands_target.txt'
    commands = './data_txt/commands.txt'
    with open(commands, 'r') as f:
        lines = f.readlines()
    num_label = len(lines)
    # base_model = FewShotKWS(num_label)
    # base_model.load_weights('/home/wuyx/kws/few_shot_kws/model/kws_test_1/checkpoints/model.030/variables/variables.index')
    base_model = tf.keras.models.load_model(model_dir)
    base_model = tf.keras.models.Sequential(base_model.layers[:-2])
    transfer_model = TransferLearn(base_model, 4)

    transfer_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    data_loader = DataLoader(target_train_list, target_dev_list, target_commands)
    data_loader.commands = ['start_recording', 'end_recording', 'unkown', 'background']
    train_loader = tf.data.Dataset.from_generator(data_loader.generator,
                                                  output_signature=(
                                                      tf.TensorSpec(shape=(None, None, 1), dtype=tf.float64),
                                                      tf.TensorSpec(shape=(), dtype=tf.int64)
                                                  ),
                                                  args=[True])
    train_loader = train_loader.shuffle(buffer_size=4000)
    train_loader = train_loader.padded_batch(8, padded_shapes=([600, 39, 1], []), drop_remainder=True)

    val_loader = tf.data.Dataset.from_generator(data_loader.generator,
                                                output_signature=(
                                                    tf.TensorSpec(shape=(None, None, 1), dtype=tf.float64),
                                                    tf.TensorSpec(shape=(), dtype=tf.int64)
                                                ),
                                                args=[False])
    val_loader = val_loader.padded_batch(4, padded_shapes=([600, 39, 1], []), drop_remainder=True)

    callbacks = [tf.keras.callbacks.CSVLogger('./export/exp_01/fold/log.csv', append=False)]

    if not os.path.exists('./export/exp_01/fold/'):
        os.makedirs('./export/exp_01/fold')

    transfer_model.fit(
        train_loader,
        validation_data=val_loader,
        steps_per_epoch=8 * 40,
        epochs=40,
        callbacks=callbacks,
        verbose=1,
        class_weight={0: 0.4, 1: 0.4, 2: 0.1, 3: 0.1},
    )

    transfer_model.backprop_into_embedding()

    transfer_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    transfer_model.fit(
        train_loader,
        validation_data=val_loader,
        steps_per_epoch=8 * 40,
        epochs=40,
        callbacks=callbacks,
        class_weight={0: 0.4, 1: 0.4, 2: 0.1, 3: 0.1},
    )

    print("saving model")
    transfer_model.save('./export/exp_01/transfer')







