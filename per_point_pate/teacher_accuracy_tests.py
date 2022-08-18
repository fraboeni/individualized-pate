from os import makedirs
from pathlib import Path
import shutil
import click
from loguru import logger
from multiprocessing import Process
import yaml

from per_point_pate.studies.parameters import ModelParameters
from per_point_pate.data.data_factory import DataFactory
from per_point_pate.studies import train_teachers

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
Tests to figure out how accurate a teacher, given a specific number of points that it is trained with
"""


def run_teacher_training(params_path, data_dir, out_dir, max_teachers=10):

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
                            n_reduce_teachers=0,
                            max_teachers=max_teachers)


def __run_teacher_training(
        params_path,
        data_dir,
        out_dir,
        averaging_count=5  # how many points to average over
):
    #model_prms = None
    accuracies = {}
    average_accuracies = {}

    train_data = None
    test_data = None
    with open(params_path, 'r') as f:
        params_dict = yaml.safe_load(f)

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

    #model_prms = prms.models
    model_prms = ModelParameters(**params_dict['models'])
    x_test, y_test = data_factory.data_test(seed=0)
    x_train, y_train = data_factory.data_private(seed=0)

    for pt_count in [200, 400, 800, 1600, 3200, 6400]:
        print(f"Pt count : {pt_count}")

        ave_accuracy = 0.
        for i in range(averaging_count):  # for averaging
            model, statistics = train_teachers.main(
                model_prms=model_prms,  #: ModelParameters,
                data_name="mnist",
                train_data=train_data,  #: Tuple[np.array],
                test_data=test_data,  #: Tuple[np.array],
                seed=0,
            )
            accuracy = statistics["test_accuracy"]
            print(f"model {i} accuracy : {accuracy}")
            ave_accuracy += accuracy
            if f"pts_{pt_count}" not in accuracies:
                accuracies[f"pts_{pt_count}"] = []
            accuracies[f"pts_{pt_count}"].append(accuracy)

        ave_accuracy = ave_accuracy / averaging_count
        print(f"Average accuracy over {pt_count} is : {ave_accuracy}")
        average_accuracies[f"pts_{pt_count}"] = ave_accuracy

    print("\n\n------\n\n")
    print("All accuracies")
    print(accuracies)

    print("Average accuracies")
    print(average_accuracies)

    # to do: pass in a yaml , that only has model parameters, and specify the input and outputs properly


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
@click.option('--max-teachers',
              '-m',
              default=10,
              help="max teachers to train.")
def main(params_path, data_dir, out_dir, max_teachers):

    run_teacher_training(params_path=params_path,
                         data_dir=data_dir,
                         out_dir=out_dir,
                         max_teachers=max_teachers)


if __name__ == "__main__":
    main()
