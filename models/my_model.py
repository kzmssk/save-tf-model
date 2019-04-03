
import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer1 = tf.keras.layers.Dense(10, activation=tf.nn.tanh)
        self.layer2 = tf.keras.layers.Dense(2, activation=tf.nn.tanh)

    def __call__(self, x):
        """ return output of this model """
        x = self.layer1(x)
        x = self.layer2(x)
        return x
