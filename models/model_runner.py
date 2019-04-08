import numpy as np
import tensorflow as tf
from .my_model import MyModel


class ModelRunner(object):
    """ helper for running MyModel """

    def __init__(self):
        # initialize model
        self.my_model = MyModel()

        # initialize placeholder of input to the model
        self.x_plh = tf.placeholder(tf.float32, shape=(1, 2))

    def forward(self):
        """ return output tensor by the model """
        return self.my_model(self.x_plh)

    def get_feed_dict(self, np_array):
        """ return feed dictionary for sess.run """
        assert np_array.shape == self.x_plh.shape
        return {self.x_plh: np_array.astype(np.float32)}

    def get_variables(self):
        """ return all variables of my_model """
        return self.my_model.get_variables()
