from copy import deepcopy
from dataclasses import dataclass
from loguru import logger
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Union
from per_point_pate.data.data_factory import DataFactory
import yaml

from per_point_pate.privacy.pate import set_duplications


@dataclass
class ResourceParameters:
    """
    Parameters regarding resources.

    This should only be used as part of an Experiment plan.

    data_dir: Parent folder for containing datasets.
        Datasets will be downloaded to this location
        if the directionry does not contian them.
    out_dir: Folder to which output will be written.
    gpu: Whether a GPU should be used.
        TODO: This might be ignored right now.
        
    """

    data_dir: Path
    out_dir: Path
    gpu: bool

    def __post_init__(self):
        self.data_dir = self.data_dir.resolve()
        self.out_dir = self.out_dir.resolve()


@dataclass
class DataParameters:
    """
    Parameters regarding the used dataset.

    This should only be used as part of an Experiment plan.

    data_name: Name of the dataset that will be used for the experiment.
        Available dataset names depend on the per_point_pate.data.DataFactory class.
    n: Total number of data points in the the used dataset.
    n_test: Number of data points that should be used as test set.
    n_public: Number of data points that should be used as public data set.
    """
    data_name: str
    n: int
    n_test: int
    n_public: int

    @property
    def n_private(self):
        """
        Number of data points that are used as private data set.
        """
        return self.n - self.n_test - self.n_public


@dataclass
class ModelParameters:
    """
    Parameters regarding the model architecture and training.

    This should only be used as part of an Experiment plan.

    architecture: Model architecture used for teachers and students, see per_point_pate.models.
        TODO: Currently, only pytorch models are usable, see per_point_pate.models.pytorch.pytorch_models.ARCHITECTURES
    teacher_epochs: Number of epochs teacher models are trained.
    student_epochs: Number of epochs student models are trained.
    lr: Used learning rate.
    weight_decay: Weight decay during training. 
        TODO: Not necessarily used.
    """
    architecture: str
    teacher_epochs: int
    student_epochs: int
    lr: float
    weight_decay: float


@dataclass
class PATEParameters:
    """
    PATE parameters defining single experiment.

    Objects of this class should be derived through the ExperimentPlan.derive_experiment_parameters 
    and never be created manually.
    """
    seed: int
    seeds2: List[int]
    n_teachers: int
    sigma: int
    sigma1: int
    t: int
    delta: float
    aggregators: List[str]
    collector: str
    budgets: List[float]
    n_labels: int
    limits: List[Union[str, List[float]]]
    distribution: Dict[Union[int, str], List[float]]
    precision: float
    epsilon_weights: np.array

    @property
    def eps_short(self):
        return [round(x, 2) for x in self.budgets]


@dataclass
class ExperimentParameters:
    resources: ResourceParameters
    data: DataParameters
    models: ModelParameters
    pate: PATEParameters

    @property
    def teachers_dir(self):
        path = self.resources.out_dir / 'teachers'
        path = path / f'{self.pate.n_teachers}_teachers_{self.pate.eps_short}_{self.pate.distribution}'
        path = path / f'seed_{self.pate.seed}'
        path = path / f'{self.pate.collector}'

        return path

    def voting_dir(self, voting_seed: int):
        path = self.resources.out_dir / 'voting'
        path = path / f'{self.pate.n_teachers}_teachers_{self.pate.eps_short}_{self.pate.distribution}'
        path = path / f'seed_{self.pate.seed}_{voting_seed}'
        return path

    def voting_output_path(self, voting_seed: int, aggregator: str):
        filename = f'{aggregator}_{self.pate.collector}_{self.pate.delta}_{self.pate.sigma}_{self.pate.sigma1}_{self.pate.t}_data.npz'
        return self.voting_dir(voting_seed) / filename

    def student_dir(self, voting_seed: int, aggregator: str):
        path = self.resources.out_dir / f'student'
        path = path / f'{self.pate.n_teachers}_teachers_{self.pate.eps_short}_{self.pate.distribution}'
        path = path / f'seed_{self.pate.seed}_{voting_seed}'
        path = path / f'{aggregator}_{self.pate.collector}_{self.pate.delta}_{self.pate.sigma}_{self.pate.sigma1}_{self.pate.t}_{self.pate.n_labels}'
        return path


@dataclass
class PATEExperimentPlan:
    """
    PATE parameters for an experiment plan.

    This should only be used as part of an Experiment plan.
    Upon deriving ExperimentParameters for the single experiments from an ExperimentPlan
    through calling ExperimentPlan.derive_experiment_parameters,
    the correct PATEParameters are derived for each experiment.

    seeds: List of seeds used for initializing randomness for a run.
        This includes shuffling data for teacher training and initializing weights.
        n seeds increase the number of derived concrete executions of the PATE pipeline n times.
        TODO: Weights initialization does not seem to be fixed yet.
    seeds2: List of seeds used for shuffling public data before teacher votes.
        TODO: In the code, the alias voting_seed is used in most places already, renaming should be finished.
    nums_teachers: List with numbers of teachers to be trained.
        With upsamping PATE, these numbers are incread to reache correct ratios for points with different privacy levels.
        The lists given for parameters 'nums_teachers', 'sigmas', 'sigma1s', 'ts' and 'deltas' have to have the same lengths.
        Passing list with length n increases the number of derived concrete executions of the PATE pipeline n times.
    sigma: List of scales of noise added to the result of each vote.
        The lists given for parameters 'nums_teachers', 'sigmas', 'sigma1s', 'ts' and 'deltas' have to have the same lengths.
        Passing list with length n increases the number of derived concrete executions of the PATE pipeline n times.
    sigma1: List of scales of noise used in the confident-GNMax to check teacher consent.
        The lists given for parameters 'nums_teachers', 'sigmas', 'sigma1s', 'ts' and 'deltas' have to have the same lengths.
        Passing list with length n increases the number of derived concrete executions of the PATE pipeline n times.
    ts: List of thresholds for confident-GNMax to check teacher consent.
        The lists given for parameters 'nums_teachers', 'sigmas', 'sigma1s', 'ts' and 'deltas' have to have the same lengths.
        Passing list with length n increases the number of derived concrete executions of the PATE pipeline n times.
    deltas: List of delta prameters for the DP guarantees.
        The lists given for parameters 'nums_teachers', 'sigmas', 'sigma1s', 'ts' and 'deltas' have to have the same lengths.
        Passing list with length n increases the number of derived concrete executions of the PATE pipeline n times.
    aggregators: Aggregators e.g. 'confident' used in PATE pipeline.
        TODO: Currently, only 'default' and 'confident' seem to be available.
    budgets: Mapping between collector type and lists of budgets .
        Available collectors are 'uGNMax', 'vGNMax', 'wGNMax', representing the different schemes to personalize pate.
        For each collector, a list of budget lists has to be specified.
        The ratio of points assigned to these budgets different budgets is described thourgh the 'distribution' parameter.
        Adding n colletors with m lists specifying budgets the number of derived concrete executions of the PATE pipeline n*m times.
    n_labels: Number of labels that should be produced during the voting process.
        n_labels > 0 - voting will stop if n_labels where produced.
        n_labels == 0 - voting will stop when the available budgets are reached.
        n_labels < 0 - the complete public dataset will be labeled.
    limits: List of given limits, determines the number of labels used for student training.
        'budgets' - The maximum number of labels with regards to the privacy budget is selected from the voted labels and used for student training.
        float - Specifies a budget independently from the attribute 'budgets',
                the maximum number of labels with regards to the privacy budget is selected from the voted labels and used for student training.
        int - Specifies a fixed number of labels selected for student training.
        TODO: I think this specification for 'limits' is correct, though only 'budgets' was actually used so far.
    distribution: Specifies the ratio of data points per label that is assigned to the groups with different privacy requirements given with 'budgets'.
    precision: Only used for uGNMax collector. Precision with which the ratio between privacy groups is mapped to the number of teachers.
    """
    seeds: List[int]
    seeds2: List[int]
    nums_teachers: List[int]
    sigmas: List[int]
    sigmas1: List[int]
    ts: List[int]
    deltas: List[float]
    aggregators: List[str]
    budgets: Dict[str, List[List[float]]]
    n_labels: int
    limits: List[Union[str, List[float]]]
    distributions: List[Dict[Union[int, str], List[float]]]
    precision: float


@dataclass
class ExperimentPlan:
    """
    Plan specifying multiple experiments.

    Call ExperimentPlan.derive_experiment_parameters to derive objects of class ExperimentParameters,
    each specifying one execution of the PATE pipeline.
    The 'resources', 'data' and 'model' parameters will be the same for all derived experiments.
    """

    resources: ResourceParameters
    data: DataParameters
    models: ModelParameters
    pate: PATEExperimentPlan

    @staticmethod
    def load(params_path: Path, data_dir: Path, out_dir: Path):
        with open(params_path, 'r') as f:
            params_dict = yaml.safe_load(f)
            return ExperimentPlan(
                resources=ResourceParameters(data_dir=data_dir,
                                             out_dir=out_dir,
                                             **params_dict['resources']),
                data=DataParameters(**params_dict['data']),
                models=ModelParameters(**params_dict['models']),
                pate=PATEExperimentPlan(**params_dict['pate']))

    def _scale_for_uGNMax(
        self,
        pate_prms: PATEParameters,
        n_per_label: Dict,
    ):
        # adjust budgets to match ratio for GNMax
        duplications = set_duplications(budgets=np.array(pate_prms.budgets),
                                        precision=self.pate.precision)

        # scale up number of teachers
        if len(pate_prms.distribution.keys()) == 1:
            n_teachers_scaled = int(
                sum(np.array(duplications) * np.array(pate_prms.distribution))
                * pate_prms.n_teachers)
        else:
            n_teachers_scaled = int(
                round(
                    sum([
                        sum(np.array(duplications) * np.array(ratios)) *
                        n_per_label[label] * pate_prms.n_teachers /
                        self.data.n_private
                        for label, ratios in pate_prms.distribution.items()
                    ])))

        # uGNMax f scales hyperparams according to new number of teachers
        # rounding just convention
        f = n_teachers_scaled / pate_prms.n_teachers
        pate_prms.sigma = round(f * pate_prms.sigma, 3)
        pate_prms.sigma1 = round(f * pate_prms.sigma1, 3)
        pate_prms.t = round(f * pate_prms.t, 3)
        pate_prms.n_teachers = n_teachers_scaled

        return pate_prms

    def _simplify_distribution(self, distribution):
        v0 = list(distribution.values())[0]
        if all([v == v0 for v in list(distribution.values())]):
            return {'all': v0}

    def derive_experiment_parameters(self, data_factory: DataFactory):
        """
        Call this to derive objects of class ExperimentParameters,
        each specifying one execution of the PATE pipeline.
        """
        experiments_params = []

        for seed in self.pate.seeds:
            n_per_label = data_factory.n_private_per_label(seed)

            for n_teachers, sigma, sigma1, t, delta in zip(
                    self.pate.nums_teachers, self.pate.sigmas,
                    self.pate.sigmas1, self.pate.ts, self.pate.deltas):
                for collector in self.pate.budgets:
                    for eps in self.pate.budgets[collector]:
                        for distribution in self.pate.distributions:

                            epsilon_weights = np.zeros(len(eps))
                            for label in distribution.keys():
                                epsilon_weights += n_per_label[
                                    label] * np.array(distribution[label])
                            epsilon_weights /= sum(epsilon_weights)

                            pate_prms = PATEParameters(
                                seed=seed,
                                seeds2=deepcopy(self.pate.seeds2),
                                n_labels=self.pate.n_labels,
                                n_teachers=n_teachers,
                                sigma=sigma,
                                sigma1=sigma1,
                                t=t,
                                delta=delta,
                                aggregators=deepcopy(self.pate.aggregators),
                                collector=collector,
                                budgets=deepcopy(eps),
                                limits=deepcopy(self.pate.limits),
                                distribution=deepcopy(distribution),
                                precision=self.pate.precision,
                                epsilon_weights=epsilon_weights,
                            )

                            # only for uGNMax
                            # precision for findig ratio of teachers for budgets
                            if collector == 'uGNMax':
                                pate_prms = self._scale_for_uGNMax(
                                    pate_prms, n_per_label)

                            # epsilon weights needed if distribution given per label
                            # calculates ratio of data points for lower:higher budget independent of label
                            pate_prms.distribution = self._simplify_distribution(
                                distribution)

                            experiments_params.append(
                                ExperimentParameters(
                                    resources=deepcopy(self.resources),
                                    data=deepcopy(self.data),
                                    models=deepcopy(self.models),
                                    pate=pate_prms))

        return experiments_params
