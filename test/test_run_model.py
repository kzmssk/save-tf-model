import pytest
import numpy as np
import tensorflow as tf
from models.my_model import MyModel

def test_run_forward():
    my_model = MyModel()

    x = tf.placeholder(tf.float32, shape=(1, 2))
    y = my_model(x)

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    x_data = np.array([[0.5, 0.6]]).astype(np.float32)
    y_data = sess.run(y, {x: x_data})

    assert y_data.shape == (1, 2)


    
