import ast
from os import makedirs
import numpy as np
import pandas as pd
from loguru import logger
from per_point_pate.experiments.experiment_factory import ExperimentFactory

from per_point_pate.studies.parameters import ExperimentParameters
from per_point_pate.data.data_factory import DataFactory


def main(
    prms: ExperimentParameters,
    data_factory: DataFactory,
):
    """
    This method trains a set of students as part of the PATE pipeline.

    For each previously executed voting, one student is trained on
    the public part of the dataset using the created labels.

    @param experiment_class: Class defining the training of one single student.
    @param prms: Parameters for the experiment,
        used for the training of all students.
    """

    train_student_fn = ExperimentFactory(prms.data.data_name).step_student

    combinations = [(voting_seed, aggregator)
                    for voting_seed in prms.pate.seeds2
                    for aggregator in prms.pate.aggregators]

    for i, (voting_seed, aggregator) in enumerate(combinations):
        student_dir = prms.student_dir(voting_seed=voting_seed,
                                       aggregator=aggregator)
        if (student_dir / 'model.h5').is_file() \
            and (student_dir / 'model.pickle').is_file():
            logger.info(
                f"Train student for voting_seed: {voting_seed}, aggregator: {aggregator}"
                f"has already been trained.")
            continue

        logger.info(
            f"Train student for voting_seed: {voting_seed}, aggregator: {aggregator}"
        )

        x_test, y_test = data_factory.data_test(seed=prms.pate.seed)
        voting_data = np.load(
            prms.voting_output_path(voting_seed=voting_seed,
                                    aggregator=aggregator))
        features = voting_data['features']
        y_pred = voting_data['y_pred']
        y_true = voting_data['y_true']

        # load voting cost curve
        path_stats_voting = str(prms.resources.out_dir / 'stats_votings.csv')
        stats_voting = pd.read_csv(path_stats_voting, header=0)
        costs_curve = np.array(
            ast.literal_eval(stats_voting[
                (stats_voting['seed'] == prms.pate.seed)
                # & (stats_voting['seed2'] == voting_seed) &
                & (stats_voting['voting_seed'] == voting_seed) &
                (stats_voting['aggregator'] == aggregator) &
                (stats_voting['collector'] == prms.pate.collector) &
                (stats_voting['eps_short'] == str(prms.pate.eps_short)) &
                (stats_voting['distribution'] == str(prms.pate.distribution)) &
                (stats_voting['n_teachers'] == prms.pate.n_teachers) &
                (stats_voting['delta'] == prms.pate.delta) &
                (stats_voting['sigma'] == prms.pate.sigma) &
                (stats_voting['sigma1'] == prms.pate.sigma1) &
                (stats_voting['t'] == prms.pate.t)].iloc[0]['costs_curve']))

        if prms.pate.limits == ['budgets']:
            limits = [prms.pate.budgets]
        else:
            limits = prms.pate.limits
        for limit in limits:
            if isinstance(limit, int):
                n_limit = sum(np.cumsum(y_pred != -1) <= limit)
                n_labels = limit
            else:
                n_limit = sum(np.all(costs_curve <= np.array(limit), axis=1))
                n_labels = n_limit - sum(y_pred[:n_limit] == -1)
            costs = costs_curve[n_limit - 1]

            if n_labels < 2000:
                # select samples for which teachers responded
                response_filter = y_pred != -1
                x_train = features[response_filter][:n_labels]
                y_train = y_pred[response_filter][:n_labels]

                student, statistics = train_student_fn(
                    prms=prms,
                    train_data=(x_train, y_train),
                    test_data=(x_test, y_test),
                )

                # save student and statistics together with parameters
                student_dir = prms.student_dir(voting_seed=voting_seed,
                                               aggregator=aggregator)
                makedirs(student_dir, exist_ok=True)
                student.save(path=student_dir)
                stats_path = prms.resources.out_dir / 'stats_students.csv'
                statistics.update(prms.pate.__dict__)
                statistics.update({
                    'voting_seed': voting_seed,
                    'aggregator': aggregator,
                    'costs': costs,
                })
                pd.DataFrame(data=[statistics.values()],
                             columns=statistics.keys()).to_csv(
                                 path_or_buf=stats_path,
                                 mode='a',
                                 header=not stats_path.is_file())
