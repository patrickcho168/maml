import tensorflow as tf
import numpy as np

class Task():
    def get_train_batch(self, k):
        raise NotImplementedError()

    def get_val_batch(self, k):
        raise NotImplementedError()

    def get_test_batch(self, k):
        raise NotImplementedError()

    def get_test_set(self):
        raise NotImplementedError()

class Dataset():
    def sample_train_tasks(self, m):
        raise NotImplementedError()

    def sample_val_tasks(self, m):
        raise NotImplementedError()

    def sample_test_tasks(self, m):
        raise NotImplementedError()

def build_meta_model(n_inputs, n_outputs, hyperparams, model, backproped_model, loss_fn):
    # Step 1: Read Hyperparameters
    # Note: If meta_gradient_steps == 0, normal transfer learning
    N = hyperparams['N'] # Number of tasks sampled
    meta_gradient_steps = hyperparams['meta_gradient_steps'] # number of meta gradient steps to take
    max_gradient_steps = hyperparams['max_gradient_steps'] # max number of gradient steps to eventually take
    alpha = hyperparams['alpha'] # task learning rate
    beta = hyperparams['beta'] # meta learning rate
    clip_grads = hyperparams['clip_grads'] # boolean to indicate whether or not to clip gradients
    assert N > 0
    assert meta_gradient_steps <= max_gradient_steps
    meta_model = {}
    
    # Step 2: Set up input and output placeholders
    # Note: We need to set up N of them as we input N tasks per meta-training step
    # Note: We also use the first one for meta-testing
    # Note: we use train to do gradient descent with K examples and val for calculating the metaloss
    x_train_phs = []
    y_train_phs = []
    x_val_phs = []
    y_val_phs = []
    for i in range(N):
        train_x = tf.placeholder(tf.float32, shape=[None, n_inputs])
        train_y = tf.placeholder(tf.float32, shape=[None, n_outputs])
        val_x = tf.placeholder(tf.float32, shape=[None, n_inputs])
        val_y = tf.placeholder(tf.float32, shape=[None, n_outputs])
        x_train_phs.append(train_x)
        y_train_phs.append(train_y)
        x_val_phs.append(val_x)
        y_val_phs.append(val_y)
        if i == 0:
            meta_model['x_train_ph'] = train_x
            meta_model['y_train_ph'] = train_y
            meta_model['x_val_ph'] = val_x
            meta_model['y_val_ph'] = val_y
    meta_model['x_train_phs'] = x_train_phs
    meta_model['y_train_phs'] = y_train_phs
    meta_model['x_val_phs'] = x_val_phs
    meta_model['y_val_phs'] = y_val_phs
    
    original_scope = 'original'
    meta_loss = None
    inner_loss = None
    train_grad_step_losses = []
    val_grad_step_losses = []
    val_y_hats = []
    for i in range(N):
        with tf.variable_scope(original_scope, reuse=(i!=0)): 
            
            # Step 3: Set up model and task loss function
            train_y_hat = model(x_train_phs[i], reuse=(i!=0))
            train_loss = loss_fn(train_y_hat, y_train_phs[i])
            val_y_hat = model(x_val_phs[i], reuse=True)
            val_loss = loss_fn(val_y_hat, y_val_phs[i])
            if i == 0:
                train_grad_step_losses.append(train_loss)
                val_grad_step_losses.append(val_loss)
                val_y_hats.append(model(meta_model['x_val_ph'], reuse=True))
                meta_model['original_loss'] = val_loss
                if meta_gradient_steps == 0: # No meta learning
                    meta_model['test_meta_loss'] = val_loss
                    meta_model['test_inner_loss'] = train_loss
                    meta_loss = val_loss
                    inner_loss = train_loss
            else:
                if meta_gradient_steps == 0: # No meta learning
                    meta_loss += val_loss
                    inner_loss += train_loss
            
            # Step 4: Set up meta learning framework
            # Note: We continue for max_gradient_steps to show that further updates improve the loss
            params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=original_scope)
            for j in range(max_gradient_steps):
                with tf.variable_scope('model_{}_gradient_iteration{}'.format(i, j)):
                    grads = tf.gradients(train_loss, params)
                    params = [param - alpha*grad for (param, grad) in zip(params, grads)]
                    new_train_y_hat = backproped_model(x_train_phs[i], params)
                    train_loss = loss_fn(new_train_y_hat, y_train_phs[i])
                    new_val_y_hat = backproped_model(x_val_phs[i], params)
                    val_loss = loss_fn(new_val_y_hat, y_val_phs[i])
                    if i == 0:
                        train_grad_step_losses.append(train_loss)
                        val_grad_step_losses.append(val_loss)
                        val_y_hats.append(backproped_model(meta_model['x_val_ph'], params))
                if j == meta_gradient_steps - 1:
                    if i == 0:
                        meta_loss = val_loss
                        inner_loss = train_loss
                        meta_model['test_meta_loss'] = val_loss
                        meta_model['test_inner_loss'] = train_loss
                    else:
                        meta_loss += val_loss
                        inner_loss += train_loss
    meta_loss /= N
    inner_loss /= N

    # Step 5: Set up meta-training optimizer
    optimizer = tf.train.AdamOptimizer(beta)
    if clip_grads:
        gvs = optimizer.compute_gradients(meta_loss)
        gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(gvs)
    else:
        train_op = optimizer.minimize(meta_loss)

    meta_model['meta_loss'] = meta_loss
    meta_model['inner_loss'] = inner_loss
    meta_model['train_op'] = train_op
    meta_model['train_grad_step_losses'] = train_grad_step_losses
    meta_model['val_grad_step_losses'] = val_grad_step_losses
    meta_model['val_y_hats'] = val_y_hats
    return meta_model

def solve_maml(n_inputs, n_outputs, hyperparams, model, backproped_model, loss_fn, dataset, 
               tensorboard_logger, test_callback=None):
    tf.reset_default_graph()
    meta_model = build_meta_model(n_inputs, n_outputs, hyperparams, model, backproped_model, loss_fn)
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)

        # Meta Training
        best_val_meta_loss = np.inf
        best_itr = 0
        for itr in range(hyperparams['meta_itrs']):
            train_tasks = dataset.sample_train_tasks(hyperparams['N'])
            val_tasks = dataset.sample_val_tasks(hyperparams['N'])
            # For Meta Training
            feed_dict = {}
            for j, task in enumerate(train_tasks):
                train_X, train_Y = task.get_train_batch(hyperparams['train_K'])
                val_X, val_Y = task.get_val_batch(hyperparams['val_K'])
                feed_dict[meta_model['x_train_phs'][j]] = train_X
                feed_dict[meta_model['y_train_phs'][j]] = train_Y
                feed_dict[meta_model['x_val_phs'][j]] = val_X
                feed_dict[meta_model['y_val_phs'][j]] = val_Y
            _, train_inner_loss, train_meta_loss = sess.run([meta_model['train_op'], meta_model['inner_loss'], 
                                                             meta_model['meta_loss']], feed_dict=feed_dict)
            # For Meta Validation
            feed_dict = {}
            for j, task in enumerate(val_tasks):
                train_X, train_Y = task.get_train_batch(hyperparams['train_K'])
                val_X, val_Y = task.get_val_batch(hyperparams['val_K'])
                feed_dict[meta_model['x_train_phs'][j]] = train_X
                feed_dict[meta_model['y_train_phs'][j]] = train_Y
                feed_dict[meta_model['x_val_phs'][j]] = val_X
                feed_dict[meta_model['y_val_phs'][j]] = val_Y
            val_inner_loss, val_meta_loss = sess.run([meta_model['inner_loss'], meta_model['meta_loss']], feed_dict=feed_dict)
            if val_meta_loss < best_val_meta_loss:
                best_val_meta_loss = val_meta_loss
                best_itr = itr
                if hyperparams['early_stopping'] > 0 and itr - best_itr > hyperparams['early_stopping']:
                    # TODO: Change params back to those at best_itr
                    break
            tensorboard_logger.log_scalar(train_inner_loss, 'train_loss/inner_loss')
            tensorboard_logger.log_scalar(train_meta_loss, 'train_loss/meta_loss')
            tensorboard_logger.log_scalar(val_inner_loss, 'val_loss/inner_loss')
            tensorboard_logger.log_scalar(train_meta_loss, 'val_loss/meta_loss')
            if itr % 200 == 0:
                print('Iteration {}: Train Inner Loss={}, Val Inner Loss={}, Train Meta Loss={}, Val Meta Loss={}'.format(
                    itr, train_inner_loss, val_inner_loss, train_meta_loss, val_meta_loss))
        
        # Meta Testing
        total_test_original_loss = 0
        total_test_task_loss = 0
        all_grad_step_losses = []
        for itr in range(hyperparams['test_itrs']):
            test_task = dataset.sample_test_tasks(1)[0]
            train_X, train_Y = test_task.get_train_batch(hyperparams['train_K'])
            val_X, val_Y = test_task.get_test_set()
            feed_dict[meta_model['x_train_ph']] = train_X
            feed_dict[meta_model['y_train_ph']] = train_Y
            feed_dict[meta_model['x_val_ph']] = val_X
            feed_dict[meta_model['y_val_ph']] = val_Y
            original_loss, task_loss, grad_step_losses, original_y_hat, one_step_y_hat, final_y_hat = \
                                        sess.run([meta_model['original_loss'],
                                                  meta_model['test_meta_loss'],
                                                  meta_model['val_grad_step_losses'],
                                                  meta_model['val_y_hats'][0],
                                                  meta_model['val_y_hats'][1],
                                                  meta_model['val_y_hats'][-1]], feed_dict=feed_dict)
            if test_callback is not None:
                test_callback(itr, test_task, original_y_hat, one_step_y_hat, final_y_hat)
            total_test_original_loss += original_loss
            total_test_task_loss += task_loss
            all_grad_step_losses.append(grad_step_losses)
        mean_test_original_loss = total_test_original_loss / hyperparams['test_itrs']
        mean_test_task_loss = total_test_task_loss / hyperparams['test_itrs']
        mean_grad_step_losses = np.mean(np.array(all_grad_step_losses), axis=0)
        print('Original Mean Loss: {}'.format(mean_test_original_loss))
        print('After {} Gradient Step(s) Mean Loss: {}'.format(hyperparams['meta_gradient_steps'], mean_test_task_loss))
        print('Mean Loss Over Gradient Steps: {}'.format(mean_grad_step_losses))
        return mean_grad_step_losses

