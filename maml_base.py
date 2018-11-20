class Task():
    def get_ground_truth(self, x):
        raise NotImplementedError()

    def get_examples(self, k):
        raise NotImplementedError()

class Dataset():
    def sample_tasks(self, m):
        raise NotImplementedError()