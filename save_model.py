import tensorflow as tf
import numpy as np

from models.my_model import MyModel
from test.test_utils import init_model_vars
from utils import get_export_dir


def save_model():
    export_dir = get_export_dir()

    my_model = MyModel()

    x = tf.placeholder(tf.float32, shape=(1, 2), name="input")
    y = my_model(x)

    with tf.Session() as sess:
        init_model_vars(sess, my_model)

        result = sess.run(y, {x: np.array([[1, 2]], dtype=np.float32)})
        print(f"y = {result}")

        tf.saved_model.simple_save(
            sess,
            str(export_dir),
            inputs={"inputs": x},
            outputs={"outputs": y}
        )


if __name__ == "__main__":
    save_model()
