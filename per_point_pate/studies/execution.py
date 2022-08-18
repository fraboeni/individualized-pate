from os import cpu_count
from time import time
from multiprocessing import Lock, Process, Semaphore


class Executor:

    def __init__(self,
                 experiments: list,
                 n_cores: int = None,
                 n_queued: int = None,
                 verbose: bool = True):
        """
        This class defines a process manager to enable parallel CPU execution.
        @param experiments: list that contains all experiments to be executed
        @param n_cores:     int to specify the number of CPU cores to use
        @param n_queued:    int to specify the number of experiments to be queued together
        @param verbose:     bool to specify if the executor prints updates after each group of queued experiments
        """
        self.experiments = experiments
        self.semaphore = Semaphore(
            max(1, min(cpu_count() -
                       1, n_cores)) if n_cores is not None else cpu_count() -
            1)
        self.n_queued = max(100, min(
            100, n_queued)) if n_queued is not None else 100
        self.verbose = verbose
        self.total = len(experiments)
        self.lock = Lock()
        self.n_started = 0
        self.current_processes = []
        self.start_time = 0

    def execute_process(self, i):
        """
        This method creates processes from experiments, and starts them.
        @param i:   int to specify the number of the experiment to start.
        @return:    None
        """
        p = Process(target=self.experiments[i].run_concurrently,
                    args=(self.semaphore, self.lock))
        self.current_processes.append(p)
        p.start()

    def runtime(self):
        """
        This method computes the difference between the current time and the time the executor was started.
        @return:    string that contains the current runtime
        """
        current = time() - self.start_time
        hours = int(current / 60**2)
        current -= hours * 60**2
        minutes = int(current / 60)
        current -= minutes * 60
        seconds = int(current)
        return f'{hours}h:{minutes}m:{seconds}s'

    def execute(self):
        """
        This method manages all experiments and executes them within a semaphore to make use of multiple CPU cores.
        Further, it uses a lock to avoid the experiments to edit the same result file at the same time.
        @return:    None
        """
        self.start_time = time()
        while self.n_started < self.total:
            if self.verbose:
                print(
                    f'{self.n_started}/{self.total} experiments ran in time {self.runtime()}.'
                )
            for i in range(self.n_started,
                           min(self.n_started + self.n_queued, self.total)):
                self.execute_process(i)
            self.n_started += self.n_queued
            for p in self.current_processes:
                p.join()
            self.current_processes.clear()
        if self.verbose:
            print(
                f'{self.total}/{self.total} experiments ran in time {self.runtime()}.'
            )
