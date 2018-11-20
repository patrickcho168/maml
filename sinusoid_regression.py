import numpy as np
import tensorflow as tf
from maml_base import Task, Dataset, build_meta_model
from models import FCN_generator, MSE_generator

'''
From paper:
Amplitude range [0.1, 5.0]
Phase range [0, pi]
x range [-5,5]
y = Asin(x+C) where A is amplitude and C is phase
Loss is MSE
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


def solve_sinusoid(hyperparams):
    # Fixed Hyperparameters
    tf.reset_default_graph()
    hidden_units = [40, 40]
    activations = [tf.nn.relu, tf.nn.relu]

    # Iterations
    meta_itrs = 10000
    test_itrs = 100

    # Task Specific Variables
    n_inputs = 1
    n_outputs = 1
    amplitude_min = 0.1
    amplitude_max = 5.0
    phase_min = 0
    phase_max = np.pi
    x_min = -5
    x_max = 5

    dataset = SinusoidDataset(amplitude_min, amplitude_max, phase_min, phase_max, x_min, x_max)
    model, backproped_model = FCN_generator(hidden_units, activations, n_outputs)
    loss_fn = MSE_generator()
    meta_model = build_meta_model(n_inputs, n_outputs, hyperparams, model, backproped_model, loss_fn)
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)

        # Meta Training
        for i in range(meta_itrs):
            tasks = dataset.sample_tasks(hyperparams['N'])
            feed_dict = {}
            for j, task in enumerate(tasks):
                X, Y = task.get_examples(hyperparams['K'])
                feed_dict[meta_model['x_phs'][j]] = X
                feed_dict[meta_model['y_phs'][j]] = Y
            _, meta_loss = sess.run([meta_model['train_op'], meta_model['meta_loss']], feed_dict=feed_dict)
            if i % 200 == 0:
                print('Iteration {}: Meta Loss={}'.format(i, meta_loss))
        
        # Meta Testing
        total_test_original_loss = 0
        total_test_task_loss = 0
        all_grad_step_losses = []
        for i in range(test_itrs):
            task = dataset.sample_tasks(1)[0]
            X, Y = task.get_examples(hyperparams['K'])
            feed_dict[meta_model['x_ph']] = X
            feed_dict[meta_model['y_ph']] = Y
            original_loss, task_loss, grad_step_losses = sess.run([meta_model['original_loss'], meta_model['task_loss'], 
                                                                   meta_model['grad_step_losses']], feed_dict=feed_dict)
            total_test_original_loss += original_loss
            total_test_task_loss += task_loss
            all_grad_step_losses.append(grad_step_losses)
            # print('Original Loss={}, Loss After {} Gradient Step(s)={}'.format(original_loss, hyperparams['meta_gradient_steps'], task_loss))
        mean_test_original_loss = total_test_original_loss / test_itrs
        mean_test_task_loss = total_test_task_loss / test_itrs
        mean_grad_step_losses = np.mean(np.array(all_grad_step_losses), axis=0)
        print('Original Mean Loss: {}'.format(mean_test_original_loss))
        print('After {} Gradient Step(s) Mean Loss: {}'.format(hyperparams['meta_gradient_steps'], mean_test_task_loss))
        print('Mean Loss Over Gradient Steps: {}'.format(mean_grad_step_losses))
        return mean_grad_step_losses

if __name__ == '__main__':
    hyperparams = {
        'K': 10,
        'alpha': 0.01,
        'beta': 0.01,
        'N': 10,
        'meta_gradient_steps': 0,
        'max_gradient_steps': 10,
    }
    results = []
    for meta_gradient_steps in range(11):
        hyperparams['meta_gradient_steps'] = meta_gradient_steps
        results.append(solve_sinusoid(hyperparams))
    for i, result in enumerate(results):
        print(result)

