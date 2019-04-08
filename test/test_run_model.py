import pytest
import numpy as np
import tensorflow as tf
from models.model_runner import ModelRunner


def test_run_forward():
    model_runner = ModelRunner()
    y = model_runner.forward()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        x_data = np.array([[0.5, 0.6]]).astype(np.float32)
        y_data = sess.run(y, model_runner.get_feed_dict(x_data))

        assert y_data.shape == (1, 2)
