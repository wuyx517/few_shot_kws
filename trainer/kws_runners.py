import os
import tensorflow as tf


class KwsTrainer:
    def __init__(self, dataloader, val_loader, model):
        self.dataloader = dataloader
        self.val_loader = val_loader
        self.model = model
        self.loss_f = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()
        self.checkpoint_dir = './model/kws_test/checkpoints/'
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_dir,
            mode="max",
        )
        self.steps = 0

    def set_train_metrics(self):
        self.train_metrics = {
            "loss": tf.keras.metrics.Mean("loss", dtype=tf.float32)
        }

    def compile(self):
        # self.model._build([1, 600, 39, 1])
        self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss_f,
                metrics=["accuracy"],
                )
        try:
            self.load_checkpoint()
        except:
            print('trainer resume failed')
        # self.model.summary()


    def save_checkpoint(self, max_save=10):
        """Save checkpoint."""
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.model.save_weights(os.path.join(self.checkpoint_dir, 'model_{}.h5'.format(self.steps)))
        print('Successfully Saved Checkpoint')
        if len(os.listdir(self.checkpoint_dir)) > max_save:
            files = os.listdir(self.checkpoint_dir)
            files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
            os.remove(os.path.join(self.checkpoint_dir, files[0]))

    def load_checkpoint(self):
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.model.load_weights(os.path.join(self.checkpoint_dir, files[-1]))
        self.steps = int(files[-1].split('_')[-1].replace('.h5', ''))

    def _train_step(self):
        self.model.fit(
            self.dataloader,
            validation_data=self.val_loader,
            epochs=40,
            callbacks=[self.model_checkpoint_callback]
            # callbacks=[tf.keras.callbacks.EarlyStopping(verbose=1, patience=4),
            #            tf.keras.callbacks.LearningRateScheduler(scheduler)],
        )

        # for bach in self.dataloader:

            # features, label = bach
            # with tf.GradientTape() as tape:
            #     y_pred = self.model(features)
            #     loss = self.loss_f(label, y_pred)
            #     # tape.watch(loss)
            # # print(self.model.trainable_variables)
            # gradients = tape.gradient(loss, self.model.trainable_variables)
            # self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            # # self.train_metrics["loss"].update_state(loss)
            # print('steeps: ', self.steps, 'loss: ', loss)
            # self.steps += 1




