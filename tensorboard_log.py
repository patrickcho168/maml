from tensorboardX import SummaryWriter
import numpy as np
import os
from collections import defaultdict

class TensorboardXLogger():
    def __init__(self, log_dir=None): # TODO: make xport_every much larger
        if not log_dir:
            log_dir = "runs"
        self.writer = SummaryWriter(log_dir)

        self.name_to_iters = defaultdict(int)

    def log_scalar(self, val, name):
        n_itr = self.name_to_iters[name]
        self.name_to_iters[name] += 1
        self.writer.add_scalar(name, val, n_itr)

    def log_returns(self, paths, name):
        undiscounted_returns = [sum(path["rewards"]) for path in paths]

        number_paths = len(undiscounted_returns)
        average_return = np.mean(undiscounted_returns)
        max_return = np.max(undiscounted_returns)
        min_return = np.min(undiscounted_returns)

        self.log_scalar(number_paths, "{}/number_paths".format(name))
        self.log_scalar(average_return, "{}/average_return".format(name))
        self.log_scalar(max_return, "{}/max_return".format(name))
        self.log_scalar(min_return, "{}/min_return".format(name))

    def log_loss(self, name, n_itr, loss, gradients=None):
        self.log_scalar("{}/loss".format(name), loss, n_itr)
        if gradients:
            self.log_scalar("{}/grad_norm".format(name), np.linalg.norm(gradients), n_itr)
