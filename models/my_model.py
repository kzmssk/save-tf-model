
import tensorflow as tf


class MyModel(tf.keras.Model):

    scope = 'my_model'

    def __init__(self):
        super().__init__()

        self.layer1 = tf.keras.layers.Dense(10, activation=tf.nn.tanh)
        self.layer2 = tf.keras.layers.Dense(2, activation=tf.nn.tanh)

        self.input_name = None
        self.output_name = None

    def __call__(self, x):
        """ return output of this model """
        self.input_name = x.name
        with tf.name_scope(self.scope):
            x = self.layer1(x)
            x = self.layer2(x)
            x = tf.identity(x, name="output")
        self.output_name = x.name
        return x

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

    def get_io_names(self):
        assert not self.input_name is None and not self.output_name is None, "input_name or output_name is not set"
        return self.input_name, self.output_name
