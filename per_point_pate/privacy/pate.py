from pathlib import Path
import math
from typing import Dict, List, Tuple, Union
import numpy as np
import os
from loguru import logger

from per_point_pate.models.classifiers import Classifier, encode_one_hot
from per_point_pate.privacy.core import compute_logq_gaussian, _log1mexp
from tqdm import tqdm


def average_dp_budgets(epsilons, deltas, weights=None):
    n = len(epsilons)
    assert n == len(deltas)
    if weights:
        assert n == len(weights)
        #assert sum(weights) == 1
    else:
        weights = [1 / n] * n
    epsilons_avg = math.log(sum(
        np.exp(np.array(epsilons)) * np.array(weights)))
    deltas_avg = sum(np.array(deltas) * np.array(weights))
    return epsilons_avg, deltas_avg


def set_duplications(budgets, precision):
    """
    This method is used for upsampling PATE only. It determines the number of duplications each sensitive data point
    gets corresponding to the given budgets.
    @param budgets:     list or array that contains one privacy budget as epsilon-value for each sensitive data point
    @param precision:   float to determine how precisely the intended ratios between different privacy budgets have to
                        be approximated
    @return:            array of integers to determine the number of duplications for each point
    """
    # precision is, when you compute a real-valued number, how close to an integer to round to, for number-of-duplications.
    p = int(1 / precision)
    budgets = np.around(np.array(budgets / np.min(budgets)) * p) / p
    tmp = np.copy(budgets)
    while np.any(np.abs(np.around(tmp, 2) - np.around(tmp)) > precision):
        tmp += budgets
    return np.around(tmp).astype(int)


def pack_overlapping_subsets(data_private: Tuple[np.array, np.array],
                             n_teachers: int, budgets_per_sample: np.array,
                             seed: int, precision: float):
    """
    This method is used for upsampling PATE only. It determines the allocation of all sensitive data points
    (including duplications) to all teachers.
    @param data_private: tuple of features and labels
    @param n_teachers:   int for the number of teachers to be trained
    @param budgets:      list of budgets corresponding to the private data
    @param seed:         int for the random seeding
    @param precision:    float to determine how precisely the intended ratios between different privacy budgets have to
                         be approximated
    @return:             tuple that contains the subsets of sensitive data (including duplications) for all teachers, and
                         a list that contains each a list of all data points corresponding to one teacher for each
                         teacher
    """
    x_private, y_private = data_private
    n_data = len(y_private)
    assert len(budgets_per_sample) == n_data

    duplications = set_duplications(budgets=budgets_per_sample,
                                    precision=precision)
    mapping_t2p = [[] for _ in range(n_teachers)]
    teacher = 0
    np.random.seed(seed)
    permutation = np.random.permutation(n_teachers)
    for point in range(n_data):
        for _ in range(min(n_teachers, duplications[point])):
            mapping_t2p[permutation[teacher]] += [point]
            teacher += 1
            if teacher == n_teachers:
                teacher = 0
                permutation = np.random.permutation(n_teachers)
    subsets = []
    for points in mapping_t2p:
        subsets.append((x_private[points], y_private[points]))
    return subsets, mapping_t2p


def pack_subsets(data_private: Tuple[np.array, np.array],
                 n_teachers: int,
                 budgets_per_sample: np.array,
                 seed: int,
                 collector: str,
                 precision: float = 0.1):
    """ This method determines the allocation of all sensitive data to all teachers.
    @param data_private: tuple of features and labels
    @param n_teachers:   int for the number of teachers to be trained
    @param budgets:      list of budgets corresponding to the private data
    @param seed:         int for the random seeding
    @param collector:    Collector that is used for personalized privacy accounting
    @param precision:    float to determine how precisely the intended ratios between different privacy budgets have to
                         be approximated
    @return:             tuple that contains the subsets of sensitive data (including duplications) for all teachers, and
                         a list that contains each a list of all data points corresponding to one teacher for each
                         teacher
    """
    assert collector in ['GNMax', 'uGNMax', 'vGNMax', 'wGNMax']
    if collector == 'uGNMax':
        return pack_overlapping_subsets(data_private=data_private,
                                        n_teachers=n_teachers,
                                        budgets_per_sample=budgets_per_sample,
                                        seed=seed,
                                        precision=precision)

    x_private, y_private = data_private
    n_data = len(y_private)
    assert len(budgets_per_sample) == n_data

    if collector == 'GNMax':
        budgets_per_sample = np.array([min(budgets_per_sample)] * n_data)
    levels = np.unique(budgets_per_sample)
    ratios = np.array(
        [sum(budgets_per_sample == level) / n_data for level in levels])
    assert all(ratios >= 1 / n_teachers - 0.01)
    nums_partitions = np.array(
        [int(round(ratio * n_teachers, 0)) for ratio in ratios])
    nums_partitions[np.argmax(
        nums_partitions)] += n_teachers - sum(nums_partitions)
    subsets = []
    mapping_t2p = []
    np.random.seed(seed)
    for i, n_partitions in enumerate(nums_partitions):
        idx_level = np.array(budgets_per_sample == levels[i])
        idx_used = np.array([False] * sum(idx_level))
        n_data_teacher = sum(idx_level) // n_partitions
        for j in range(n_partitions):
            features_tmp = x_private[idx_level][np.logical_not(idx_used)]
            labels_tmp = y_private[idx_level][np.logical_not(idx_used)]
            int_idx = np.random.choice(a=range(len(labels_tmp)), size=n_data_teacher, replace=False) \
                if j < n_partitions - 1 else np.arange(len(labels_tmp), dtype=int)
            idx_teacher = np.array([False] * len(labels_tmp))
            idx_teacher[int_idx] = True
            mapping_t2p.append(
                list(
                    np.arange(len(budgets_per_sample))[idx_level][
                        np.logical_not(idx_used)][idx_teacher]))
            x_train = features_tmp[
                idx_teacher] if j < n_partitions - 1 else features_tmp
            y_train = labels_tmp[idx_teacher]
            idx_used[np.logical_not(idx_used)] = np.logical_or(
                idx_used[np.logical_not(idx_used)], idx_teacher)
            subsets.append((x_train, y_train))
    return subsets, mapping_t2p


def set_budgets_per_sample(
    y_private: np.array,
    budgets: List[float],
    distribution: Dict[Union[str, int], List[float]],
    seed: int,
):
    """
    This method determines the privacy specification of each sensitive data point according to the given distribution of
    privacy specifications.
    @param y_private:       array of all labels of the sensitive data
    @param budgets:         list of different privacy specifications
    @param distribution:    dict that contains one tuple for each label which contains one ratio for each epsilon
    @param seed:            seed that determines the pseudo-random number generation
    @return:                array that contains the privacy budget of each sensitive data point
    """
    np.random.seed(seed)

    n_budgets = len(budgets)
    n_private = len(y_private)
    assert all([
        len(distribution[label]) == n_budgets for label in distribution.keys()
    ])
    assert all(
        [sum(distribution[label]) == 1 for label in distribution.keys()])

    budget_per_sample = np.ones(n_private)
    for i, label in enumerate(distribution.keys()):

        # true case expects 'all', check this
        if label in y_private:
            idx = np.arange(n_private)[y_private == label]
        elif label == 'all':
            idx = np.arange(n_private)
        else:
            raise RuntimeError(f"Unknown label in distrubution: {label}.")

        n = len(idx)
        for j, ratio in enumerate(distribution[label]):
            if ratio == 0:
                continue
            # if it is the last ratio, do last
            if j == n_budgets - 1:
                budget_per_sample[idx] *= budgets[j]
            else:
                # all examples with label, sample from them
                selected_idx = np.random.choice(idx,
                                                int(round(n * ratio)),
                                                replace=False)
                budget_per_sample[selected_idx] *= budgets[j]
                # remove used indices
                idx = np.setdiff1d(idx, selected_idx)

    return budget_per_sample


def save_mappings(
    budgets_per_sample: np.array,
    mapping_t2p: np.array,
    teachers_dir: Path,
):
    os.makedirs(teachers_dir, exist_ok=True)
    np.save(teachers_dir / 'budgets_per_sample.npy', budgets_per_sample)
    np.save(teachers_dir / 'mappings_t2p.npy', mapping_t2p, allow_pickle=True)


def load_mappings(teachers_dir: Path, ):
    budgets_per_sample = np.load(teachers_dir / 'budgets_per_sample.npy')
    mapping_t2p = np.load(teachers_dir / 'mappings_t2p.npy', allow_pickle=True)
    return budgets_per_sample, mapping_t2p


def rdp_to_dp(alphas, epsilons, delta):
    """
    This method computes the DP costs for a given delta corresponding to given RDP costs.
    @param alphas:      list of RDP orders
    @param epsilons:    list of RDP costs corresponding to alphas
    @param delta:       float that specifies the precision of DP costs
    @return:            tuple that contains the computed DP costs, and the corresponding best RDP order
    """
    assert len(alphas) == len(epsilons)
    eps = np.array(epsilons) - math.log(delta) / (np.array(alphas) - 1)
    idx_opt = np.argmin(eps)
    return eps[idx_opt], alphas[idx_opt]


class PATE:

    PRECISION = 0.05

    def __init__(
        self,
        seed: int,
        n_teachers,
        #  teachers: List[Classifier],
        n_classes: int,
        #  unlabeled_data: np.ndarray,
        predictions: np.ndarray,
        budgets: np.ndarray,
        mapping: list,
        aggregator_name: str,
        collector_name: str,
        delta: float,
        alphas: np.ndarray,
        sigma: float,
        n_labels: int,
        sigma1: float = None,
        t: int = None,
    ):
        """
        Objects of this class handle the privacy-preserving creation of labels for unlabeled public data using teacher
        models trained on sensitive data.

        @param seed:            seed for pseudo random number generator to create reproducible randomness
        @param n_teachers:      number of teachers
        @param teachers:        list of classifiers that were trained on sensitive data or path to their directory
        @param n_classes:       number of classes corresponding to the data distribution
        @param unlabeled_data:  features of public data to be labeled by PATE voting
        @param budgets:         array of one privacy budget per sensitive data point used to train teachers
        @param mapping:         list of each one list of all data points used to train one teacher
        @param aggregator_name: method to perturb the aggregate of teachers' votes and to select which labels are taken;
                                    currently only 'default' and 'confident' are supported
        @param collector_name:  method to collect teachers' votes and to account the (individual) privacy cost;
                                    currently only 'GNMax','uGNMax', 'vGNMax', and 'wGNMax' are supported
        @param delta:           magnitude of uncertainty of privacy costs; recommendation: delta << 1/(#sensitive data)
        @param alphas:          list of floats to determine the orders of RDP-bounds to specify privacy costs
        @param sigma:           standard deviation of noise added to vote aggregates to enforce privacy
        @param n_labels:        number of labels to be generated by votings; if 0: votings stop as budgets are exhausted
        @param sigma1:          standard deviation of noise used to select labels for 'confident' aggregator
        @param t:               consensus threshold to select labels for 'confident' aggregator
        """
        # self.n_teachers = len(teachers)
        self.n_teachers = n_teachers
        self.seed = seed
        # self.teachers = teachers
        self.n_classes = n_classes
        # self.X = unlabeled_data
        self.predictions = predictions
        self.budgets = budgets
        self.mapping_t2p = mapping
        self.aggregator_name = aggregator_name
        self.collector_name = collector_name
        self.delta = delta
        self.alphas = alphas
        self.sigma = sigma
        self.n_labels = n_labels
        self.sigma1 = sigma1
        self.t = t
        self.aggregator = None
        self.ratios = []
        self.labels = []
        self.rdp_costs = None
        self.dp_costs = []
        self.simple_budgets = []
        self.simple_costs = []
        self.alpha_history = []

    def prepare(self):
        """
        This method creates the collector and aggregator. It further creates empty lists or arrays to prepare for the
        privacy accounting.
        @return: None
        """
        collectors = {
            'GNMax': GNMax,
            'uGNMax': UpsamplingGNMax,
            'vGNMax': VanishingGNMax,
            'wGNMax': WeightingGNMax
        }
        aggregators = ['default', 'confident']
        levels = np.unique(self.budgets)
        assert self.collector_name in collectors
        assert self.aggregator_name in aggregators
        assert len(levels) <= self.n_teachers
        assert all(self.alphas > 1)
        c = collectors[self.collector_name](n_classes=self.n_classes,
                                            n_teachers=self.n_teachers,
                                            sigma=self.sigma,
                                            seed=self.seed)
        c.prepare(budgets=self.budgets, mapping=self.mapping_t2p)
        if self.aggregator_name == 'default':
            self.aggregator = DefaultGNMax(collector=c, seed=self.seed)
        if self.aggregator_name == 'confident':
            self.aggregator = ConfidentGNMax(collector=c,
                                             seed=self.seed,
                                             sigma=self.sigma1,
                                             t=self.t)
        self.ratios = [sum(self.budgets == level) for level in levels]
        self.rdp_costs = np.zeros(
            (len(self.aggregator.collector.groups), len(self.alphas)))

    def predict_all(self):
        """
        This method creates labels for the unlabeled public data. It stops if each data point has been voted for, if
        the number of requested labels (n_labels) is reached, or if n_labels = 0 and any privacy budget is exhausted.
        For each voting, the privacy cost per group (all data points that have the same budget, or all teachers
        corresponding to the same budget in vGNMax) is tracked and cumulated by RDP for each order in alphas. Moreover,
        the RDP costs are temporarily converted into (epsilon, delta)-DP costs by selecting the order that results in
        the minimal costs. For the confident aggregator, the non-selected labels are stored as -1.
        @return: None
        """
        count_labels = 0
        count_votings = 0
        pbar = tqdm()
        # as long as less votes than predictions made by each teacher
        while count_votings < np.shape(self.predictions)[1]:
            if count_labels == self.n_labels:
                break
            label, costs = self.aggregator.vote(
                alphas=self.alphas,
                mapping=self.mapping_t2p,
                predictions=self.predictions[:, count_votings])
            tmp_rdp_costs = self.rdp_costs
            tmp_rdp_costs[self.aggregator.collector.mapping_c2g] += costs
            tmp_dp_costs_alpha = [
                rdp_to_dp(alphas=self.alphas,
                          epsilons=tmp_rdp_costs[group_budget],
                          delta=self.delta) for group_budget in range(
                              len(self.aggregator.collector.budgets_g))
            ]
            tmp_dp_costs = np.array([cost for cost, _ in tmp_dp_costs_alpha])
            tmp_alphas = [alpha for _, alpha in tmp_dp_costs_alpha]
            if self.n_labels == 0 and any(
                    self.aggregator.collector.budgets_g < tmp_dp_costs):
                break
            self.rdp_costs = tmp_rdp_costs
            self.dp_costs.append(tmp_dp_costs)
            self.alpha_history.append(tmp_alphas)
            self.labels.append(label)
            count_labels += 1 if label != -1 else 0
            count_votings += 1
            pbar.set_description(desc=f'votes: {count_votings}')

    def simplify_budgets_costs(self):
        """
        This method rounds each value in budgets and in the cost history. Moreover, it simplifies both s.t. there are
        only as many budgets and costs per label as there are different privacy levels.
        @return: None
        """
        decimals = 5
        levels = np.unique(self.budgets)
        p = 1 / PATE.PRECISION
        levels = np.around(np.array(levels) * p) / p
        self.simple_budgets = list(levels)
        if self.collector_name == 'GNMax':
            self.simple_budgets = [min(self.budgets)]
        self.simple_costs = [[round(cost, decimals) for cost in costs]
                             for costs in self.dp_costs]
        if self.collector_name == 'vGNMax':
            self.simple_costs = [[
                round(
                    max(costs[np.arange(
                        len(self.aggregator.collector.budgets_g))[
                            self.aggregator.collector.budgets_g == level]]),
                    decimals) for level in levels
            ] for costs in self.dp_costs]
            """
            self.simple_costs = []
            for costs in self.dp_costs:
                for level in levels:      
                    budgets_to_find_max = costs[np.arange(
                            len(self.aggregator.collector.budgets_g))[
                                self.aggregator.collector.budgets_g == level]]
                    if len(budgets_to_find_max) >0:
                        max_budget = round( max(budgets_to_find_max), decimals)
                    else: 
                        max_budget = 0
                self.simple_costs.append(max_budget)
            """


class Collector:
    def __init__(self, n_classes: int, n_teachers: int, sigma: float,
                 seed: int):
        """
        This is an abstract class to determine methods to collect teachers' votes and to account (individual) privacy
        costs.

        @param n_classes:   number of classes corresponding to the data distribution
        @param n_teachers:  number of classifiers that were trained on sensitive data
        @param sigma:       standard deviation of noise added to vote aggregates to enforce privacy
        @param seed:        seed for pseudo random number generator to create reproducible randomness
        """
        self.n_classes = n_classes
        self.sigma = sigma
        self.seed = seed
        self.prng = np.random.RandomState(seed)
        self.n_teachers = n_teachers
        self.mapping_c2p = None
        self.groups = None
        self.budgets_g = None
        self.mapping_c2g = None

    def prepare(self, budgets, mapping):
        """
        @param budgets:     list of one (epsilon, delta)-DP epsilon per sensitive data point
        @param mapping:     list of lists: each inner list contains indices of all data points learned by one teacher
        @return:            None
        """
        raise NotImplementedError

    def collect_votes(self, mapping, predictions):
        raise NotImplementedError

    def tight_bound(self, votes, alphas):
        raise NotImplementedError

    @staticmethod
    def loose_bound_personalized(sigma, alphas, sensitivities):
        raise NotImplementedError

    @staticmethod
    def loose_bound(sigma, alphas):
        raise NotImplementedError


class GNMax(Collector):
    def __init__(self, n_classes: int, n_teachers: int, sigma: float,
                 seed: int):
        """
        This class implements the Gaussian NoisyMax (GNMax) mechanism from Papernot et al.: 'Scalable Private Learning
        with PATE' 2018.

        @param n_classes:   number of classes corresponding to the data distribution
        @param n_teachers:  number of classifiers that were trained on sensitive data
        @param sigma:       standard deviation of noise added to vote aggregates to enforce privacy
        @param seed:        seed for pseudo random number generator to create reproducible randomness
        """
        super().__init__(n_classes=n_classes,
                         n_teachers=n_teachers,
                         sigma=sigma,
                         seed=seed)
        self.sensitivities = None

    def prepare(self, budgets, mapping):
        self.budgets_g = np.array([min(budgets)])
        self.sensitivities = np.array([1])
        self.groups = np.array([0])
        self.mapping_c2g = self.groups

    def collect_votes(self, mapping, predictions):
        votes = np.zeros(self.n_classes)

        for i in range(self.n_classes):
            votes[i] = sum(predictions == i)

        return votes

    def tight_bound(self, votes, alphas):
        epsilons = np.zeros((len(self.sensitivities), len(alphas)))
        for i, sensitivity in enumerate(self.sensitivities):
            sigma = self.sigma / sensitivity
            vote_counts = votes / sensitivity
            log_q = compute_logq_gaussian(counts=vote_counts, sigma=sigma)
            assert log_q <= 0 < sigma and np.all(alphas > 1)
            if np.isneginf(log_q):
                return np.full_like(alphas, 0., dtype=np.float)
            alpha2 = math.sqrt(sigma**2 * -log_q)
            alpha1 = alpha2 + 1
            epsilons[i] = self.loose_bound(alphas=alphas, sigma=sigma)
            mask = np.logical_and(alpha1 > np.atleast_1d(alphas), alpha2 > 1)
            epsilons1 = self.loose_bound(alphas=[alpha1], sigma=sigma)
            epsilons2 = self.loose_bound(alphas=[alpha2], sigma=sigma)
            log_a2 = (alpha2 - 1) * epsilons2
            if (np.any(mask) and log_q <= log_a2 - alpha2 *
                (math.log(1 + 1 / (alpha1 - 1)) + math.log(1 + 1 /
                                                           (alpha2 - 1)))
                    and -log_q > epsilons2):
                log1q = _log1mexp(log_q)
                log_a = (alphas - 1) * (log1q - _log1mexp(
                    (log_q + epsilons2) * (1 - 1 / alpha2)))
                log_b = (alphas - 1) * (epsilons1 - log_q / (alpha1 - 1))
                log_s = np.logaddexp(log1q + log_a, log_q + log_b)
                epsilons[i][mask] = np.minimum(epsilons[i],
                                               log_s / (alphas - 1))[mask]
            assert np.all(np.array(epsilons[i]) >= 0)
        return epsilons

    @staticmethod
    def loose_bound_personalized(sigma, alphas, sensitivities):
        return np.array([[s**2 * alpha / sigma**2 for alpha in alphas]
                         for s in sensitivities])

    @staticmethod
    def loose_bound(sigma, alphas):
        return np.array([alpha / sigma**2 for alpha in alphas])


class UpsamplingGNMax(GNMax):
    def __init__(self, n_classes: int, n_teachers: int, sigma: float,
                 seed: int):
        """
        This class implements a personalized extension of the Gaussian NoisyMax (GNMax) mechanism from Papernot et al.:
        'Scalable Private Learning with PATE' 2018. This extension is able to account each an individual privacy cost
        per sensitive data point depending on the number of teachers that were trained on it. Therefore, it is called
        upsampling Gaussian NoisyMax (uGNMax).

        @param n_classes:   number of classes corresponding to the data distribution
        @param n_teachers:  number of classifiers that were trained on sensitive data
        @param sigma:       standard deviation of noise added to vote aggregates to enforce privacy
        @param seed:        seed for pseudo random number generator to create reproducible randomness
        """
        super().__init__(n_classes=n_classes,
                         n_teachers=n_teachers,
                         sigma=sigma,
                         seed=seed)

    def prepare(self, budgets, mapping):
        self.budgets_g = np.unique(budgets)
        duplications = np.zeros(len(budgets))
        for teacher_points in mapping:
            for p in teacher_points:
                duplications[p] += 1
        duplications_unique = np.unique(duplications)
        self.groups = np.array(range(len(duplications_unique)))
        self.mapping_c2g = self.groups
        self.sensitivities = duplications_unique


class VanishingGNMax(GNMax):
    def __init__(self, n_classes: int, n_teachers: int, sigma: float,
                 seed: int):
        """
        This class implements a personalized extension of the Gaussian NoisyMax (GNMax) mechanism from Papernot et al.:
        'Scalable Private Learning with PATE' 2018. This extension is able to account each an individual privacy cost
        per sensitive data point depending on the voting participation of the teacher that was trained on it. Therefore,
        it is called vanishing Gaussian NoisyMax (vGNMax).

        @param n_classes:   number of classes corresponding to the data distribution
        @param n_teachers:  number of classifiers that were trained on sensitive data
        @param sigma:       standard deviation of noise added to vote aggregates to enforce privacy
        @param seed:        seed for pseudo random number generator to create reproducible randomness
        """
        super().__init__(n_classes=n_classes,
                         n_teachers=n_teachers,
                         sigma=sigma,
                         seed=seed)
        self.probs = None
        self.precision = PATE.PRECISION
        self.participation = None
        self.freqs = None
        self.freqs_unique = None
        self.cycle_len = 0
        self.n_votings = 0

    def prepare(self, budgets, mapping):
        self.groups = np.array(range(self.n_teachers))
        self.budgets_g = [
            min(budgets[mapping[t]]) for t in range(self.n_teachers)
        ]

        p = 1 / self.precision
        self.budgets_g = np.around(np.array(self.budgets_g) * p) / p
        probs = np.array(self.budgets_g) / max(self.budgets_g)
        self.probs = probs / max(probs)
        tmp_probs = np.copy(probs)
        f = 1
        while np.any(
                np.abs(np.around(tmp_probs, 2) - np.around(tmp_probs)) > 0):
            f += 1
            tmp_probs = f * probs
        self.freqs = np.around(tmp_probs).astype(int)
        self.freqs_unique = np.unique(self.freqs)
        self.cycle_len = max(self.freqs_unique)

    def collect_votes(self, mapping, predictions):
        n_cycle = self.n_votings % self.cycle_len
        if n_cycle == 0:
            self.participation = self.determine_participation()
        self.n_votings += 1
        self.mapping_c2g = self.participation[n_cycle]
        weight = self.n_teachers / sum(self.mapping_c2g)
        self.sensitivities = np.array([weight])

        if False:
            print("!\n" * 10)
            print(self.sensitivities)
            import sys
            sys.exit(0)

        votes = np.zeros(self.n_classes)

        for i in range(self.n_classes):
            votes[i] = sum(predictions[self.mapping_c2g] == i)

        return votes * weight

    def determine_participation(self):
        participation = np.zeros((self.cycle_len, len(self.freqs)))
        rest = 0
        switch1 = np.zeros(self.cycle_len, dtype=bool)
        switch2 = np.zeros(self.cycle_len, dtype=bool)
        for freq in self.prng.permutation(self.freqs_unique):
            idx = self.freqs == freq
            n_freq = sum(idx)
            if freq == self.cycle_len:
                participation[:, idx] = 1
            else:
                mask = self.prng.permutation(np.arange(n_freq))
                num = n_freq * freq / self.cycle_len
                t = 0
                for i in range(self.cycle_len):
                    if switch1[i]:
                        if not switch2[i]:
                            rest = 0.5
                            switch1[i] = False
                            switch2[i] = True
                        else:
                            rest = 0
                            switch2[i] = False
                    tmp_participation = np.zeros(n_freq)
                    tmp_num = round(num + rest)
                    rest = num - tmp_num + rest
                    switch1[i] = rest >= 0.25
                    j = 0
                    if t == n_freq:
                        mask = self.prng.permutation(mask)
                    while j < tmp_num:
                        if t == n_freq:
                            t = 0
                        if tmp_participation[mask[t]] == 0:
                            tmp_participation[mask[t]] += 1
                            j += 1
                        t += 1
                    participation[i, idx] = tmp_participation
        return participation.astype(bool)


class WeightingGNMax(GNMax):
    def __init__(self, n_classes: int, n_teachers: int, sigma: float,
                 seed: int):
        """
        This class implements a personalized extension of the Gaussian NoisyMax (GNMax) mechanism from Papernot et al.:
        'Scalable Private Learning with PATE' 2018. This extension is able to account each an individual privacy cost
        per sensitive data point depending on the influence weight on voting of the teacher that was trained on it.
        Therefore, it is called weighted Gaussian NoisyMax (wGNMax).

        @param n_classes:   number of classes corresponding to the data distribution
        @param n_teachers:  number of classifiers that were trained on sensitive data
        @param sigma:       standard deviation of noise added to vote aggregates to enforce privacy
        @param seed:        seed for pseudo random number generator to create reproducible randomness
        """
        super().__init__(n_classes=n_classes,
                         n_teachers=n_teachers,
                         sigma=sigma,
                         seed=seed)
        self.weights = None

    def prepare(self, budgets, mapping):

        # mapping is a list going from teacher to weight
        ###     # probably a list of lists, each teacher-list holds all the points a teacher is trained on

        # budgets is a list, of length data points
        #       n_data = len(y_private)
        #       assert len(budgets_per_sample) == n_data

        self.budgets_g = np.unique(budgets)
        budgets_t = np.array(
            [min(budgets[mapping[t]]) for t in range(self.n_teachers)])

        # budgets is length of teachers

        epsilons, counts = np.unique(budgets_t, return_counts=True)
        # new weighting per teacher is : (distribution_i * (scaled-epsilon_i))

        weighting = (counts / sum(counts)) * (
            epsilons /
            (sum(epsilons)))  # np.array of length <number-of-distributions>

        weighting_d = {epsilons[i]: weighting[i] for i in range(len(epsilons))}

        weighting_2 = np.array(
            [weighting_d[budgets_t[t]] for t in range(self.n_teachers)])

        # budgets_t = np.log(1 + budgets_t) # this is budgets per teacher (list)

        #self.weights = budgets_t / sum(budgets_t) * self.n_teachers
        self.weights = weighting_2 / sum(weighting_2) * self.n_teachers

        assert round(sum(self.weights)) == self.n_teachers
        weights_unique = np.unique(self.weights)
        self.groups = np.array(range(len(weights_unique)))
        self.mapping_c2g = self.groups
        self.sensitivities = weights_unique

    def collect_votes(self, mapping, predictions=None):
        votes = np.zeros(self.n_classes)

        for i in range(self.n_classes):
            votes[i] = sum((predictions == i).astype(int) * self.weights)
        return votes


class Aggregator:
    def __init__(self, collector: Collector, seed: int):
        """
        This is an abstract class to determine methods to perturb the aggregate of teachers' votes and to select which
        labels are taken.

        @param collector:   method to collect teachers' votes and to account the (individual) privacy cost
        @param seed:        seed for pseudo random number generator to create reproducible randomness
        """
        self.collector = collector
        self.n_classes = collector.n_classes
        self.seed = seed
        self.prng = np.random.RandomState(seed)

    def aggregate_votes(self, votes, alphas):
        """
        @param votes:   array that contains the vote count of each class
        @param alphas:  array of floats > 1 that determine the orders of RDP bounds
        @return:        label:      int that corresponds to a specific class
                        epsilons:   RDP costs (for each order in alphas) per group (sensitive data points that have the
                                    same privacy budget)
        """
        raise NotImplementedError

    def vote(self, alphas, mapping, predictions=None):
        """
        @param x:           feature to create a label for
        @param teachers:    list of classifiers that were trained on sensitive data
        @param alphas:      list of orders for calculation of RDP costs
        @param mapping:     list of teachers or data points that belong to the same group (share same privacy budget)
        @param predictions: array of teachers' predictions if PATE is terminated by reaching specific number of labels
        @return:            label:      int that corresponds to a specific class
                            epsilons:   RDP costs (for each order in alphas) per group (sensitive data points that have
                                        the same privacy budget)
        """
        votes = self.collector.collect_votes(mapping=mapping,
                                             predictions=predictions)
        return self.aggregate_votes(votes=votes, alphas=alphas)


class DefaultGNMax(Aggregator):
    def __init__(self, collector: GNMax, seed: int):
        """
        This class implements the standard aggregation method of teachers' votes that only adds Gaussian noise.

        @param collector:   method to collect teachers' votes and to account the (individual) privacy cost
        @param seed:        seed for pseudo random number generator to create reproducible randomness
        """
        super().__init__(collector=collector, seed=seed)

    def aggregate_votes(self, votes, alphas):
        """
        @param votes:   array that contains the vote count of each class
        @param alphas:  array of floats > 1 that determine the orders of RDP bounds
        @return:        label:      int that corresponds to a specific class
                        epsilons:   RDP costs (for each order in alphas) per group (sensitive data points that have the
                                    same privacy budget)
        """
        noise = self.prng.normal(loc=0.0,
                                 scale=self.collector.sigma,
                                 size=self.n_classes)
        epsilons = np.array(
            self.collector.tight_bound(votes=votes, alphas=alphas))
        label = np.argmax(votes + noise)
        return label, epsilons


class ConfidentGNMax(Aggregator):
    def __init__(self, collector: GNMax, seed: int, sigma: float, t: float):
        """
        This class implements the confident aggregation method of teachers' votes that adds Gaussian noise and selects
        labels by privately checking if there is a consensus between the teachers.

        @param collector:   method to collect teachers' votes and to account the (individual) privacy cost
        @param seed:        seed for pseudo random number generator to create reproducible randomness
        @param sigma:       standard deviation of Gaussian noise that is used to check if the teachers consent
        @param t:           threshold to determine if the teachers consent
        """
        super().__init__(collector=collector, seed=seed)
        self.sigma = sigma
        self.t = t

    def aggregate_votes(self, votes, alphas):
        """
        @param votes:   array that contains the vote count of each class
        @param alphas:  array of floats > 1 that determine the orders of RDP bounds
        @return:        label:      int that corresponds to a specific class; -1 if there is no teachers' consensus
                        epsilons:   RDP costs (for each order in alphas) per group (sensitive data points that have the
                                    same privacy budget)
        """
        noise1 = self.prng.normal(loc=0.0, scale=self.sigma, size=1)
        noise2 = self.prng.normal(loc=0.0,
                                  scale=self.collector.sigma,
                                  size=self.n_classes)
        label = -1
        epsilons = self.collector.loose_bound_personalized(
            sigma=self.sigma,
            alphas=alphas,
            sensitivities=self.collector.sensitivities) / 2
        if np.max(votes) + noise1 >= self.t:
            label = np.argmax(votes + noise2)
            epsilons += self.collector.tight_bound(votes=votes, alphas=alphas)
        return label, epsilons


class InteractiveGNMax(Aggregator):
    def __init__(self, collector: GNMax, seed: int, sigma: float, t1: float,
                 t2: float, k: int, student: Classifier):
        """
        This class implements the interactive aggregation method of teachers' votes that adds Gaussian noise, checks for
        teachers' consensus, and checks if student already knows the label.

        @param collector:   method to collect teachers' votes and to account the (individual) privacy cost
        @param seed:        seed for pseudo random number generator to create reproducible randomness
        @param sigma:       standard deviation of Gaussian noise that is used to check if the teachers consent
        @param t1:          threshold to determine if the student already knows the label
        @param t2:          threshold to determine if the teachers consent
        @param k:           factor to determine if the student already knows the label
        @param student:     classifier that should have been trained partly
        """
        super().__init__(collector=collector, seed=seed)
        self.sigma = sigma
        self.t1 = t1
        self.t2 = t2
        self.k = k
        self.student = student

    def aggregate_votes(self, votes, alphas):
        """
        @param votes:   array that contains the vote count of each class
        @param alphas:  array of floats > 1 that determine the orders of RDP bounds
        @return:        label:      int that corresponds to a specific class; -1 if there is no teachers' consensus
                        epsilons:   RDP costs (for each order in alphas) per group (sensitive data points that have the
                                    same privacy budget); if student already consents with teachers, only the costs of
                                    the consensus check are considered
        """
        x, votes = votes
        confidences = self.student.eval(x=x)
        noise1 = self.prng.normal(loc=0.0, scale=self.sigma, size=1)
        noise2 = self.prng.normal(loc=0.0,
                                  scale=self.collector.sigma,
                                  size=self.n_classes)
        label = -1
        epsilons = self.collector.loose_bound(
            sigma=self.sigma,
            alphas=alphas,
            sensitivities=self.collector.sensitivities) / 2
        if np.max(votes - self.k * confidences) + noise1 >= self.t1:
            label = np.argmax(votes + noise2)
            epsilons = self.collector.tight_bound(votes=votes, alphas=alphas)
        elif np.max_np(confidences) > self.t2:
            label = np.argmax(confidences)
        return label, epsilons
