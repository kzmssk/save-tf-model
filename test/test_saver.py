import tempfile
import tensorflow as tf
from pathlib import Path
from models.my_model import MyModel
import numpy as np

from test.test_utils import init_model_vars, get_var_sum


def test_saver():
    my_model = MyModel()
    x = tf.placeholder(tf.float32, shape=(1, 2))
    y = my_model(x)

    sess = tf.Session()

    init_model_vars(sess, my_model)

    res0 = get_var_sum(sess, my_model)

    saver = tf.train.Saver(my_model.get_variables())

    with tempfile.TemporaryDirectory() as d:
        savename = Path(str(d)) / "model"
        saver.save(sess, str(savename))
        print("===== dir contents after saver.save =====")
        for c in Path(str(d)).iterdir():
            print(c)
        print("=====================")

        init_model_vars(sess, my_model)

        res1 = get_var_sum(sess, my_model)

        assert res0 != res1, "variables are not re-initialized"
