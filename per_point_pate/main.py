from os import makedirs
from pathlib import Path
import shutil
import click
from loguru import logger
from multiprocessing import Process

from per_point_pate.studies.parameters import ExperimentPlan, ExperimentParameters
from per_point_pate.data.data_factory import DataFactory
from per_point_pate.studies import train_teachers, run_votings, train_students
"""
This file executes the complete PATE pipeline on for each parameter combination that is specified.
First, the dataset is divided into private, public, and test parts. Then, the private subset is allocated to the
teachers depending on the privacy personalization and the collector. The teachers are trained on their corresponding
subsets and stored. They are loaded afterwards for the voting that produces labels for the public subset which are
stored then. Thereafter, the labels are loaded together with the public subset to train a student. For the teachers'
and student's training as well as for the voting, statistics are stored in three different result files for subsequent
analysis.
"""


def run_pate(params_path,
             data_dir,
             out_dir,
             log_file=None,
             n_reduce_teachers=0):

    if log_file:
        logger.remove()
        logger.add(log_file)

    plan = ExperimentPlan.load(params_path=Path(params_path),
                               data_dir=Path(data_dir),
                               out_dir=Path(out_dir))

    logger.info("Starting study:")
    logger.info(plan)

    data_factory = DataFactory(data_name=plan.data.data_name,
                               data_dir=plan.resources.data_dir,
                               out_dir=plan.resources.out_dir)

    logger.info("Preparing data:")

    data_factory.write_splits(seeds=plan.pate.seeds,
                              n_test=plan.data.n_test,
                              n_public=plan.data.n_public,
                              n_private=plan.data.n_private,
                              balanced=True)

    prms: ExperimentParameters
    for prms in plan.derive_experiment_parameters(data_factory=data_factory):

        logger.info(f"Starting experiment:")
        logger.info(prms)

        logger.info(f"Training teachers:")
        train_teachers.main(prms,
                            data_factory,
                            n_reduce_teachers=n_reduce_teachers)

        run_votings.main(
            prms,
            data_factory,
        )
        logger.info(f"Training students:")
        train_students.main(
            prms,
            data_factory,
        )


@click.command()
@click.option('--params_path',
              '-p',
              required=True,
              type=click.Path(file_okay=True),
              help="Path to the parameter file.")
@click.option(
    '--data_dir',
    '-d',
    required=True,
    type=click.Path(dir_okay=True),
    help=
    "Path to the data directory (for MNIST, or whatnot - will download if data not there)."
)
@click.option('--out_dir',
              '-o',
              required=True,
              type=click.Path(dir_okay=True),
              help="Path to the output directory.")
def main(params_path, data_dir, out_dir):

    run_pate(
        params_path=params_path,
        data_dir=data_dir,
        out_dir=out_dir,
        log_file=None,
        n_reduce_teachers=0,
    )


@click.command()
@click.option('--set_params_dir',
              '-s',
              required=True,
              type=click.Path(file_okay=True))
@click.option('--data_dir',
              '-d',
              required=True,
              type=click.Path(dir_okay=True))
@click.option('--out_root_dir',
              '-o',
              required=True,
              type=click.Path(dir_okay=True))
def run_experiment_set(set_params_dir, data_dir, out_root_dir):

    set_params_dir = Path(set_params_dir).resolve()
    out_root_dir = Path(out_root_dir).resolve()
    params_paths = [
        f.resolve() for f in Path(set_params_dir).iterdir() if f.is_file()
    ]

    experiment_set_name = set_params_dir.name

    # iterate over experiment plans of set
    # and create a process for each plan
    processes = []
    for params_path in params_paths:
        experiment_plan_name = params_path.name.split('.')[0]
        # create subfolder for exeriment plan
        out_subdir = out_root_dir / experiment_plan_name
        makedirs(out_subdir, exist_ok=True)

        # copy parameter file for experiment plan
        shutil.copy(params_path, out_subdir)

        # path to logfile
        log_file = out_subdir / 'log.txt'

        # call pate
        process = Process(
            target=run_pate,
            kwargs={
                'params_path': params_path,
                'data_dir': data_dir,
                'out_dir': out_subdir,
                'log_file': log_file,
                'n_reduce_teachers': 0,
            },
            name=f'per-point-pate:{experiment_set_name}:{experiment_plan_name}'
        )
        processes.append(process)
        process.start()


if __name__ == "__main__":
    main()
