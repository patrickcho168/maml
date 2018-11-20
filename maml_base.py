import tensorflow as tf

class Task():
    def get_ground_truth(self, x):
        raise NotImplementedError()

    def get_examples(self, k):
        raise NotImplementedError()

class Dataset():
    def sample_tasks(self, m):
        raise NotImplementedError()

def build_meta_model(n_inputs, n_outputs, hyperparams, model, backproped_model, loss_fn):
    N = hyperparams['N'] # Number of tasks sampled
    K = hyperparams['K'] # Number of examples per task
    meta_gradient_steps = hyperparams['meta_gradient_steps'] # number of meta gradient steps to take
    max_gradient_steps = hyperparams['max_gradient_steps'] # max number of gradient steps to eventually take
    alpha = hyperparams['alpha'] # task learning rate
    beta = hyperparams['beta'] # meta learning rate

    assert N > 0 and K > 0 # if meta_gradient_steps == 0, normal transfer learning
    assert meta_gradient_steps <= max_gradient_steps
    meta_model = {}
    
    x_phs = []
    y_phs = []
    for i in range(N):
        x = tf.placeholder(tf.float32, shape=[None, n_inputs])
        y = tf.placeholder(tf.float32, shape=[None, n_outputs])
        x_phs.append(x)
        y_phs.append(y)
        if i == 0:
            meta_model['x_ph'] = x
            meta_model['y_ph'] = y
    meta_model['x_phs'] = x_phs
    meta_model['y_phs'] = y_phs
    
    original_scope = 'original'
    meta_loss = None
    grad_step_losses = []
    for i in range(N):
        with tf.variable_scope(original_scope, reuse=(i!=0)):   
            y_hat = model(x_phs[i])
            loss = loss_fn(y_hat, y_phs[i])
            if i == 0:
                grad_step_losses.append(loss) # 0 gradient step loss
                meta_model['y_hat'] = y_hat
                meta_model['original_loss'] = loss
                meta_model['task_loss'] = loss
                if meta_gradient_steps == 0:
                    meta_loss = loss
            else:
                if meta_gradient_steps == 0:
                    meta_loss += loss
            params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=original_scope)
            for j in range(max_gradient_steps):
                with tf.variable_scope('model_{}_gradient_iteration{}'.format(i, j)):
                    grads = tf.gradients(loss, params)
                    params = [param - alpha*grad for (param, grad) in zip(params, grads)]
                    new_y_hat = backproped_model(x_phs[i], params)
                    loss = loss_fn(new_y_hat, y_phs[i])
                    if i == 0:
                        grad_step_losses.append(loss)
                if j == meta_gradient_steps - 1:
                    if i == 0:
                        meta_loss = loss
                        meta_model['task_loss'] = loss
                    else:
                        meta_loss += loss
    # meta_loss /= N
    optimizer = tf.train.AdamOptimizer(beta)
    train_op = optimizer.minimize(meta_loss)
    meta_model['meta_loss'] = meta_loss
    meta_model['train_op'] = train_op
    meta_model['grad_step_losses'] = grad_step_losses
    return meta_model
