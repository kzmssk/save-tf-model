
import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.scope = "my_model"
        self.layer1 = tf.keras.layers.Dense(10, activation=tf.nn.tanh)
        self.layer2 = tf.keras.layers.Dense(2, activation=tf.nn.tanh)

    def __call__(self, x):
        """ return output of this model """
        with tf.name_scope(self.scope):
            x = self.layer1(x)
            x = self.layer2(x)
            x = tf.identity(x, name="output")
        return x

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
