
import tensorflow as tf


class MyModel(tf.keras.Model):

    name = 'my_model'

    def __init__(self):
        super().__init__()
        self.layer1 = tf.layers.Dense(10, activation=tf.nn.tanh)
        self.layer2 = tf.layers.Dense(2, activation=tf.nn.tanh)

    def __call__(self, x):
        """ return output of this model """
        with tf.variable_scope(self.name):
            x = self.layer1(x)
            x = self.layer2(x)
        return x

    def get_variables(self):
        """ return all variables of this model """
        res_vars = []
        for v in tf.global_variables():
            if self.name in v.name:
                res_vars.append(v)
        assert len(res_vars) > 0, 'not variable found'
        return res_vars
