import numpy as np
import tensorflow as tf
from models.my_model import MyModel


def run_model():
    my_model = MyModel()

    x = tf.placeholder(tf.float32, shape=(1, 2))
    y = my_model(x)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        x_data = np.array([[0.5, 0.6]]).astype(np.float32)
        y_data = sess.run(y, {x: x_data})

        print(y_data)

        for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="my_model"):
            print(v)


if __name__ == '__main__':
    run_model()
