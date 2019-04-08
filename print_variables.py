import tensorflow as tf
from models.model_runner import ModelRunner


def print_variables():
    model_runner = ModelRunner()

    # variables are initialized by forwarding
    y = model_runner.forward()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for v in model_runner.get_variables():
            print(f"{v}")
            print(f"sum = {sess.run(v).sum()}")


if __name__ == '__main__':
    print_variables()
