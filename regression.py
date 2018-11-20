import numpy as np
import tensorflow as tf
from maml_base import Task, Dataset

'''
Amplitude range [0.1, 5.0]
Phase range [0, pi]
x range [-5,5]
y = Asin(x+C) where A is amplitude and C is phase

Loss MSE

NN 2 hidden layers of size 40 with ReLU nonlinearities
'''

class SinusoidTask(Task):
    def __init__(self, amplitude, phase, x_min, x_max):
        self.amplitude = amplitude
        self.phase = phase
        self.x_min = x_min
        self.x_max = x_max

    def get_ground_truth(self, x):
        assert x <= self.x_max
        assert x >= self.x_min
        return self.amplitude * np.sin(x + self.phase)

    def get_examples(self, k):
        X = []
        Y = []
        for _ in range(k):
            x = np.random.uniform(self.x_min, self.x_max)
            y = self.get_ground_truth(x)
            X.append([x])
            Y.append([y])
        return np.array(X), np.array(Y)


class SinusoidDataset(Dataset):
    def __init__(self, amplitude_min, amplitude_max, phase_min, phase_max, x_min, x_max):
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.phase_min = phase_min
        self.phase_max = phase_max
        self.x_min = x_min
        self.x_max = x_max

    def sample_tasks(self, m):
        tasks = []
        for _ in range(m):
            amplitude = np.random.uniform(self.amplitude_min, self.amplitude_max)
            phase = np.random.uniform(self.phase_min, self.phase_max)
            tasks.append(SinusoidTask(amplitude, phase, self.x_min, self.x_max))
        return tasks


def build_sine_model(n_inputs, n_outputs, gradient_steps=1, alpha=0.01, beta=0.01, N=1, K=10):
    '''
    N: Number of tasks sampled
    K: Number of examples per task
    alpha: learning rate
    beta: meta learning rate
    gradient_steps: number of gradient steps to take
    '''
    assert N > 0 and K > 0 and gradient_steps > 0
    model = {}
    
    x_phs = []
    y_phs = []
    for i in range(N):
        x = tf.placeholder(tf.float32, shape=[None, n_inputs])
        y = tf.placeholder(tf.float32, shape=[None, n_outputs])
        x_phs.append(x)
        y_phs.append(y)
        if i == 0: # Placeholders for first model. Used for meta testing.
            model['x_ph'] = x
            model['y_ph'] = y
    model['x_phs'] = x_phs
    model['y_phs'] = y_phs
    
    original_scope = 'original'
    
    meta_loss = None
    for i in range(N):
        with tf.variable_scope(original_scope, reuse=(i!=0)):   
            h1 = tf.layers.dense(x_phs[i], 40, activation=tf.nn.relu)
            h2 = tf.layers.dense(h1, 40, activation=tf.nn.relu)
            y_hat = tf.layers.dense(h2, 1)
            loss = tf.reduce_mean(tf.square(y_hat - y_phs[i]))
            if i == 0:
                model['y_hat'] = y_hat
                model['original_loss'] = loss
            trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=original_scope)
            for j in range(gradient_steps):
                with tf.variable_scope('model_{}_gradient_iteration{}'.format(i, j)):
                    grads = tf.gradients(loss, trainable_variables)
                    new_params = [param - alpha*grad for (param, grad) in zip(trainable_variables, grads)]
                    new_h1 = tf.nn.relu(tf.matmul(x_phs[i], new_params[0]) + new_params[1])
                    new_h2 = tf.nn.relu(tf.matmul(new_h1, new_params[2]) + new_params[3])
                    new_y_hat = tf.matmul(new_h2, new_params[4]) + new_params[5]
                    loss = tf.reduce_mean(tf.square(new_y_hat - y_phs[i]))
            if i == 0:
                meta_loss = loss
                model['task_loss'] = loss
            else:
                meta_loss += loss
    # meta_loss /= N
    optimizer = tf.train.AdamOptimizer(beta)
    train_op = optimizer.minimize(meta_loss)
    model['meta_loss'] = meta_loss
    model['train_op'] = train_op
    return model

def solve_sinusoid():
    # Hyperparameters
    K = 10
    alpha = 0.01
    beta = 0.01
    N = 10
    gradient_steps = 1

    # Iterations
    meta_itrs = 3000
    test_itrs = 100

    # Task Specific Variables
    n_inputs = 1
    n_outputs = 1

    dataset = SinusoidDataset(0.1, 5.0, 0, np.pi, -5, 5)
    model = build_sine_model(n_inputs, n_outputs,
                             gradient_steps=gradient_steps, 
                             alpha=alpha, 
                             beta=beta, 
                             N=N, 
                             K=K)
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)

        # Meta Training
        for i in range(meta_itrs):
            tasks = dataset.sample_tasks(N)
            feed_dict = {}
            for j, task in enumerate(tasks):
                X, Y = task.get_examples(K)
                feed_dict[model['x_phs'][j]] = X
                feed_dict[model['y_phs'][j]] = Y
            _, _meta_loss = sess.run([model['train_op'], model['meta_loss']], feed_dict=feed_dict)
            print('Iteration {}: Meta Loss={}'.format(i, _meta_loss))
        
        # Meta Testing
        total_test_original_loss = 0
        total_test_task_loss = 0
        for i in range(test_itrs):
            task = dataset.sample_tasks(1)[0]
            X, Y = task.get_examples(K)
            feed_dict[model['x_ph']] = X
            feed_dict[model['y_ph']] = Y
            _original_loss, _task_loss = sess.run([model['original_loss'], model['task_loss']], feed_dict=feed_dict)
            total_test_original_loss += _original_loss
            total_test_task_loss += _task_loss
            print('Original Loss={}, Loss After One Gradient Step={}'.format(_original_loss, _task_loss))
        print('Original Mean Loss: {}'.format(total_test_original_loss/test_itrs))
        print('After One Gradient Step Mean Loss: {}'.format(total_test_task_loss/test_itrs))

if __name__ == '__main__':
    solve_sinusoid()