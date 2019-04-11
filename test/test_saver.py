import tempfile
import tensorflow as tf
from pathlib import Path
from models.my_model import MyModel
import numpy as np


def get_var_sum(sess, model):
    res = 0
    for v in model.get_variables():
        res += sess.run(v).sum()
    return res


def init_model_vars(sess, model):
    for v in model.get_variables():
        sess.run(v.initializer)


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


def test_simple_save():
    my_model = MyModel()
    x = tf.placeholder(tf.float32, shape=(1, 2), name="input")
    y = my_model(x)
    sess = tf.Session()
    init_model_vars(sess, my_model)

    with tempfile.TemporaryDirectory() as d:
        export_dir = str(Path(str(d)) / "simple_save")

        print(f"x.name = {x.name}")
        print(f"y.name = {y.name}")

        tf.saved_model.simple_save(
            sess,
            export_dir,
            inputs={"inputs": x},
            outputs={"outputs": y}
        )

        init_model_vars(sess, my_model)

        # load after save
        tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], export_dir)

        # get input and ouput again
        x_new = sess.graph.get_tensor_by_name("input:0")
        y_new = sess.graph.get_tensor_by_name("my_model_1/output:0")
        y_new_out = sess.run(
            y_new, {x_new: np.random.rand(1, 2).astype(np.float32)})
