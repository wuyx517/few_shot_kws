import keras_bert.backend
import tensorflow as tf


class FewShotKWS(tf.keras.Model):
    def __init__(self, num_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.e_net = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_tensor=None,
            pooling=None,
            input_shape=(None, None, 1)
        )
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.d1 = tf.keras.layers.Dense(1024, activation='relu')
        self.d2 = tf.keras.layers.Dense(1024, activation='relu')
        self.d3 = tf.keras.layers.Dense(192, activation='selu', kernel_initializer='lecun_normal')
        self.d4 = tf.keras.layers.Dense(num_labels)

    def _build(self, input_shape):
        features = tf.random.normal(shape=input_shape)
        self(features, trainable=False)

    def call(self, inputs, *args, **kwargs):
        output = self.e_net(inputs)
        output = self.gap(output)
        output = self.d1(output)
        output = self.d2(output)
        output = self.d3(output)
        # output = tf.reshape(inputs, [inputs.shape[0], -1])
        output = self.d4(output)
        return output

    # def call(self, inputs, training=None):
    #     output = self.gap(inputs)
    #     output = self.d1(output)
    #     output = self.d2(output)
    #     output = self.d3(output)
    #     output = self.d4(output)
    #     return output


class TransferLearn(tf.keras.Model):
    def __init__(self, embedding_model: tf.keras.Model,
                 num_label: int):
        super(TransferLearn, self).__init__()
        self.embedding_model = embedding_model
        self.embedding_model.trainable = False
        self.d1 = tf.keras.layers.Dense(units=18, activation='tanh')
        self.d2 = tf.keras.layers.Dense(units=num_label, activation='softmax')

    def backprop_into_embedding(self):
        for layer in self.embedding_model.layers[-20:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

    def call(self, inputs, training=None, mask=None):
        output = self.embedding_model(inputs)
        output = self.d1(output)
        output = self.d2(output)
        return output


if __name__ == '__main__':
    input_shape = (1, 90, 39, 1)
    few_show_kws = FewShotKWS(3)
    few_show_kws._build(input_shape)
    features = tf.random.normal(shape=input_shape)
    few_show_kws(features)





