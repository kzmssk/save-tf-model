

def get_var_sum(sess, model):
    res = 0
    for v in model.get_variables():
        res += sess.run(v).sum()
    return res


def init_model_vars(sess, model):
    for v in model.get_variables():
        sess.run(v.initializer)
