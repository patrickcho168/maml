import numpy as np
import tensorflow as tf
import datetime
import dateutil.tz
import os
import matplotlib.pyplot as plt
import pickle as pk

from maml_base import Task, Dataset, build_meta_model, solve_maml
from models import FCN_generator, MSE_generator
from tensorboard_log import TensorboardXLogger

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
        self.num_examples = 50

    def get_ground_truth(self, x):
        assert x <= self.x_max
        assert x >= self.x_min
        return self.amplitude * np.sin(x + self.phase)

    def get_train_batch(self, k):
        X = []
        Y = []
        for _ in range(k):
            x = np.random.uniform(self.x_min, self.x_max)
            y = self.get_ground_truth(x)
            X.append([x])
            Y.append([y])
        return np.array(X), np.array(Y)

    def get_val_batch(self, k):
        return self.get_train_batch(k)

    def get_test_batch(self, k):
        return self.get_train_batch(k)

    def get_test_set(self):
        X = np.linspace(self.x_min, self.x_max, self.num_examples)
        Y = np.array([self.get_ground_truth(x) for x in X])
        return X.reshape((self.num_examples, 1)), Y.reshape((self.num_examples, 1))
    
    def plot(self, *args, **kwargs):
        X = np.linspace(self.x_min, self.x_max, self.num_examples)
        Y = np.array([self.get_ground_truth(x) for x in X])
        return plt.plot(X, Y, *args, **kwargs)

    def plot_test_results(self, Y, *args, **kwargs):
        X = np.linspace(self.x_min, self.x_max, self.num_examples)
        return plt.plot(X, np.squeeze(Y), *args, **kwargs)


class SinusoidDataset(Dataset):
    def __init__(self, amplitude_min, amplitude_max, phase_min, phase_max, x_min, x_max):
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.phase_min = phase_min
        self.phase_max = phase_max
        self.x_min = x_min
        self.x_max = x_max

    def sample_train_tasks(self, m):
        tasks = []
        for _ in range(m):
            amplitude = np.random.uniform(self.amplitude_min, self.amplitude_max)
            phase = np.random.uniform(self.phase_min, self.phase_max)
            tasks.append(SinusoidTask(amplitude, phase, self.x_min, self.x_max))
        return tasks

    def sample_val_tasks(self, m):
        # Infinite dataset so can sample from same distribution
        return self.sample_train_tasks(m)

    def sample_test_tasks(self, m):
        # Infinite dataset so can sample from same distribution
        return self.sample_train_tasks(m)


def solve_sinusoid(hyperparams):
    # Model Hyperparameters
    hidden_units = [40, 40]
    activations = [tf.nn.relu, tf.nn.relu]

    # Task Specific Variables
    n_inputs = 1
    n_outputs = 1
    amplitude_min = 0.1
    amplitude_max = 5.0
    phase_min = 0
    phase_max = np.pi
    x_min = -5
    x_max = 5

    # Setting up directories and Tensorboard
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    directory = os.path.join('run', 'sinusoid', 'sinusoid_{}'.format(timestamp))
    os.makedirs(directory)
    os.makedirs(os.path.join(directory, 'plots'))
    with open(os.path.join(directory, 'hyperparams.pk'), 'wb') as f:
        pk.dump(hyperparams, f)
    tensorboard_logger = TensorboardXLogger(os.path.join(directory, 'tensorboard_log'))

    dataset = SinusoidDataset(amplitude_min, amplitude_max, phase_min, phase_max, x_min, x_max)
    model, backproped_model = FCN_generator(hidden_units, activations, n_outputs)
    loss_fn = MSE_generator()

    def test_callback(itr, train_X, train_Y, test_task, original_y_hat, one_step_y_hat, final_y_hat):
        test_task.plot(label='ground truth', color='C3')
        test_task.plot_test_results(original_y_hat, label='pre-update', color='C0')
        test_task.plot_test_results(one_step_y_hat, label='1 grad step', color='C1')
        test_task.plot_test_results(final_y_hat, label='{} grad steps'.format(hyperparams['max_gradient_steps']), color='C2')
        plt.scatter(train_X, train_Y, label='used for grad', color='C4', marker='^')
        plt.legend()
        plt.savefig(os.path.join(directory, 'plots', '{}.jpg'.format(itr)))
        plt.clf()
        plt.cla()
        plt.close()

    return solve_maml(n_inputs, n_outputs, hyperparams, model, backproped_model, loss_fn, dataset, 
                      tensorboard_logger, test_callback)

if __name__ == '__main__':
    hyperparams = {
        'train_K': 10,
        'val_K': 10,
        'alpha': 0.01,
        'beta': 0.01,
        'N': 10,
        'meta_gradient_steps': 0,
        'max_gradient_steps': 10,
        'meta_itrs': 6000,
        'test_itrs': 100,
        'clip_grads': True,
        'early_stopping': 25,
        'first_order': False,
    }
    results = []
    for meta_gradient_steps in range(5):
        hyperparams['meta_gradient_steps'] = meta_gradient_steps
        results.append(solve_sinusoid(hyperparams))
    for i, result in enumerate(results):
        print(result)

