from os.path import isfile
from pandas import DataFrame


class Experiment:

    def __init__(self, seed: int, study: str, num: int):
        """
        This class defines properties and methods that an experiment must implement.
        @param seed:    int to seed the pseudo-random number generation
        @param study:   string to specify the study the experiment belongs to
        @param num:     int to specify the number of the experiment
        """
        self.seed = seed
        self.study = study
        self.num = num
        self.result = {}

    def compute(self):
        """
        This method executes the main computation of the experiment.
        @return:    None
        """
        raise NotImplementedError

    def save(self, path=None, file_name=None):
        """
        This method stores the results of the experiment on the given location with the given name.
        @param path:        string to specify the location where the experiment is to be stored
        @param file_name:   string to specify the name of the result file
        @return:            None
        """
        path = path + '/' if path else ''
        file_name = file_name if file_name else 'results.csv'
        file = path + file_name
        DataFrame(self.result).to_csv(path_or_buf=file,
                                      index=False,
                                      mode='a',
                                      header=not isfile(file))

    def run(self):
        """
        This method runs the compute- and save-method of the experiment in a non-concurrent manner.
        @return:
        """
        self.compute()
        self.save()

    def run_concurrently(self, semaphore, lock):
        """
        This method runs the compute- and save-method of the experiment concurrently.
        @return:
        """
        with semaphore:
            self.compute()
            with lock:
                self.save()
