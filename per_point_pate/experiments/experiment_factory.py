import per_point_pate.experiments.pytorch as pytorch_experiments


class ExperimentFactory:

    def __init__(self, data_name):
        """
        Allows to get experiment classes by by name of the used dataset.

        Args:
            data_name (str): str to determine the dataset to be used (currently available: adult, mnist)

        Raises:
            RuntimeError: _description_
        """
        self.dataset_to_experiments = {
            'mnist': [
                pytorch_experiments.train_teacher,
                pytorch_experiments.vote,
                pytorch_experiments.train_student,
            ],
            'fashion_mnist': [
                pytorch_experiments.train_teacher,
                pytorch_experiments.vote,
                pytorch_experiments.train_student,
            ],
            'cifar10': [
                pytorch_experiments.train_teacher,
                pytorch_experiments.vote,
                pytorch_experiments.train_student,
            ],
            'svhn': [
                pytorch_experiments.train_teacher,
                pytorch_experiments.vote,
                pytorch_experiments.train_student,
            ],
        }

        if data_name not in self.dataset_to_experiments.keys():
            raise RuntimeError(
                f"Dataset '{data_name}' not known to ExperimentFactory.")

        self.data_name = data_name

    @property
    def step_teachers(self):
        return self.dataset_to_experiments[self.data_name][0]

    @property
    def step_voting(self):
        return self.dataset_to_experiments[self.data_name][1]

    @property
    def step_student(self):
        return self.dataset_to_experiments[self.data_name][2]
