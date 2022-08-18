from loguru import logger
import numpy as np
import os
import pandas as pd

from per_point_pate.experiments.experiment_factory import ExperimentFactory
from per_point_pate.data.data_factory import DataFactory
from per_point_pate.studies.parameters import ExperimentParameters
from per_point_pate.privacy.pate import load_mappings


def main(prms: ExperimentParameters, data_factory: DataFactory):
    """
    This method executes the voting step of the PATE pipeline
    based on a previously trained teacher ensemble.

    It executes the PATE voting on the public part of
    the dataset for each given parameter combination of voting seed and aggregator.
    Thereby, (personalized) privacy costs are tracked and
    statistics as well as the produced labels are stored afterwards.

    @param vote_fn: Function defining one teacher voting.
    @param model_type: Type of teacher models.
    @param prms: Parameters for the experiment, used for all votings.
    """
    combinations = [(voting_seed, aggregator)
                    for voting_seed in prms.pate.seeds2
                    for aggregator in prms.pate.aggregators]

    alphas = np.arange(49, dtype=float) + 2

    budgets_per_sample, mapping_t2p = load_mappings(
        teachers_dir=prms.teachers_dir)

    vote_fn = ExperimentFactory(prms.data.data_name).step_voting
    for i, (voting_seed, aggregator) in enumerate(combinations):
        voting_output_path = prms.voting_output_path(voting_seed=voting_seed,
                                                     aggregator=aggregator)
        if voting_output_path.is_file():
            logger.info(
                f"Voting for aggregator: {aggregator}, voting_seed: {voting_seed} "
                f"has already taken place.")
            continue

        logger.info(
            f"Voting for aggregator: {aggregator}, voting_seed: {voting_seed}")

        # shuffle public data according to voting_seed
        np.random.seed(voting_seed)
        x_public_data, y_public_data = data_factory.data_public(
            seed=prms.pate.seed)
        p = np.random.permutation(np.arange(len(y_public_data)))
        x_public_data = x_public_data[p]
        y_public_data = y_public_data[p]

        features, y_pred, statistics = vote_fn(
            prms=prms,
            aggregator=aggregator,
            alphas=alphas,
            public_data=(x_public_data, y_public_data),
            budgets_per_sample=budgets_per_sample,
            mapping_t2p=mapping_t2p,
        )
        voting_dir = prms.voting_dir(voting_seed=voting_seed)
        os.makedirs(voting_dir, exist_ok=True)
        assert len(y_pred) == len(features)
        y_true = y_public_data[:len(features)]
        np.savez(voting_output_path,
                 features=features,
                 y_pred=y_pred,
                 y_true=y_true)

        # save voting statistics
        statistics.update({
            'aggregator': aggregator,
            'voting_seed': voting_seed,
        })
        for key in [
                'seed', 'collector', 'eps_short', 'distribution', 'n_teachers',
                'delta', 'sigma', 'sigma1', 't'
        ]:
            statistics[key] = getattr(prms.pate, key)
        stats_path = prms.resources.out_dir / 'stats_votings.csv'
        pd.DataFrame(data=[statistics.values()],
                     columns=statistics.keys()).to_csv(
                         path_or_buf=stats_path,
                         mode='a',
                         header=not stats_path.is_file())
