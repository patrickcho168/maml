import tensorflow as tf

def FCN_generator(hidden_units, activations, n_outputs):
    hidden_units.append(n_outputs)
    activations.append(tf.identity)
    
    def FCN(x):
        h = x
        for i, (units, activation) in enumerate(zip(hidden_units, activations)):
            with tf.variable_scope('layer{}'.format(i)):
                h = tf.layers.dense(h, units, activation=activation)
        return h
    
    def backproped_FCN(x, params):
        '''
        params have to be in format (w,b,w,b,...) in order
        '''
        h = x
        for i, activation in enumerate(activations):
            h = activation(tf.matmul(h, params[i*2]) + params[i*2+1])
        return h

    return FCN, backproped_FCN

def MSE_generator():
    def loss_fn(y_hat, y):
        return tf.reduce_mean(tf.square(y_hat - y))
    return loss_fn