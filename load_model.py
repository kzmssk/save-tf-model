import tensorflow as tf
import numpy as np

from models.my_model import MyModel
from test.test_utils import init_model_vars
from utils import get_export_dir


def load_model():
    export_dir = get_export_dir()

    sess = tf.Session()

    metagraph = tf.saved_model.loader.load(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        export_dir)

    input_name = metagraph.signature_def["serving_default"].inputs["inputs"].name
    output_name = metagraph.signature_def["serving_default"].outputs["outputs"].name

    x = sess.graph.get_tensor_by_name(input_name)
    y = sess.graph.get_tensor_by_name(output_name)

    result = sess.run(y, {x: np.array([[1, 2]], dtype=np.float32)})
    print(f"y = {result}")


if __name__ == "__main__":
    load_model()
