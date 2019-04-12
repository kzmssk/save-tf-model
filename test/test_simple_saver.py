import tempfile
import tensorflow as tf
from pathlib import Path
from models.my_model import MyModel
import numpy as np

from test.test_utils import init_model_vars, get_var_sum


def _test_simple_save(tmp_dir):

    # save model variables to a directory
    graph1 = tf.Graph()
    with graph1.as_default():
        my_model = MyModel()
        x = tf.placeholder(tf.float32, shape=(1, 2), name="input")
        y = my_model(x)

        with tf.Session(graph=graph1) as sess:
            init_model_vars(sess, my_model)
            export_dir = str(Path(str(tmp_dir)) / "simple_save")
            tf.saved_model.simple_save(
                sess,
                export_dir,
                inputs={"inputs": x},
                outputs={"outputs": y}
            )
            sum1 = get_var_sum(sess, my_model)

    # launch a new graph
    graph2 = tf.Graph()
    with graph2.as_default():
        with tf.Session(graph=graph2) as sess:
            assert len(my_model.get_variables()
                       ) == 0, "there is a some variable"

            # re-init model variables
            # you can not declare x again
            y = my_model(x)
            init_model_vars(sess, my_model)

            # this should differ from sum1
            sum2 = get_var_sum(sess, my_model)
            assert sum1 != sum2, "variables are not re-initialized"

            # restore from saved model
            tf.saved_model.loader.load(
                sess, [tf.saved_model.tag_constants.SERVING], export_dir)

            # this should be same as the first sum
            sum3 = get_var_sum(sess, my_model)
            assert sum1 == sum3, "variables are not restored"


def test_simple_save():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _test_simple_save(tmp_dir)
