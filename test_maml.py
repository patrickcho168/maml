import tensorflow as tf
import numpy as np

from maml_base import build_meta_model

'''
Test Example:
a is the param
y_hat = (a^3)x
Loss = y_hat
d(y_hat)/da = 3(a^2)x
d2(y_hat)/da2 = 6ax
'''

hyperparams = {
    'train_K': 1,
    'val_K': 1,
    'alpha': 1,
    'beta': 1,
    'N': 1,
    'meta_gradient_steps': 1,
    'max_gradient_steps': 1,
    'meta_itrs': 1,
    'test_itrs': 1,
    'clip_grads': False,
    'early_stopping': 0,
    'first_order': False,
}

def get_model(init_a):
    def model(x, reuse=False):
        with tf.variable_scope('linear', reuse=reuse): 
            a = tf.get_variable(name='a', shape=(1,), initializer=tf.constant_initializer(value=init_a))
        return a*a*a*x
    return model

def backproped_model(x, params):
    a = params[0]
    return a*a*a*x

def loss_fn(y_hat, y):
    return y_hat

def da(a,x):
    return 3*a*a*x

def d2a(a,x):
    return 6*a*x

for first_order_type in [False, True]:
    for _ in range(5):
        tf.reset_default_graph()
        init_a = np.random.uniform(-5,5)
        train_x = np.random.uniform(-5,5)
        val_x = np.random.uniform(-5,5)
        alpha = np.random.uniform(0.1,1.0)
        hyperparams['alpha'] = alpha
        hyperparams['first_order'] = first_order_type
        model = get_model(init_a)
        meta_model = build_meta_model(1, 1, hyperparams, model, backproped_model, loss_fn)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            feed_dict = {}
            feed_dict[meta_model['x_train_phs'][0]] = [[train_x]]
            feed_dict[meta_model['x_val_phs'][0]] = [[val_x]]
            result = sess.run(meta_model['gvs'], feed_dict=feed_dict)[0][0][0] # First variable, gradient is first index, read a
            expected = da(init_a - alpha*da(init_a, train_x), val_x)
            if not first_order_type: # need hessian
                expected *= (1 - alpha*d2a(init_a, train_x))
            assert np.allclose(result, expected)
