from os import makedirs
from typing import Tuple
from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, TimeoutError

from datetime import datetime

import torch

from per_point_pate.studies.parameters import ExperimentParameters
from per_point_pate.data.data_factory import DataFactory
from per_point_pate.experiments.experiment_factory import ExperimentFactory
from per_point_pate.privacy.pate import pack_subsets, save_mappings, set_budgets_per_sample


def prepare_private_partitions(prms: ExperimentParameters,
                               data_private: Tuple[np.ndarray, np.ndarray]):
    paths = {
        # 'partitions': prms.teachers_dir / 'partitions.npy',
        'budgets': prms.teachers_dir / 'budgets_per_sample.npy',
        'mapping': prms.teachers_dir / 'mappings_t2p.npy'
    }
    # if all([p.is_file() for p in paths.values()]):
    #     partitions = np.load(paths['partitions'])
    #     budgets_per_sample = np.load(paths['budgets'])
    #     mapping_t2p = np.load(paths['mapping'])

    # else:

    x_private, y_private = data_private
    budgets_per_sample = set_budgets_per_sample(
        y_private=y_private,
        budgets=prms.pate.budgets,
        distribution=prms.pate.distribution,
        seed=prms.pate.seed)
    partitions, mapping_t2p = pack_subsets(
        data_private=(x_private, y_private),
        n_teachers=prms.pate.n_teachers,
        budgets_per_sample=budgets_per_sample,
        seed=prms.pate.seed,
        collector=prms.pate.collector,
        precision=prms.pate.precision,
    )
    makedirs(prms.teachers_dir, exist_ok=True)
    np.save(paths['budgets'], budgets_per_sample)
    np.save(paths['mapping'], mapping_t2p)

    return partitions, budgets_per_sample, mapping_t2p


# GLOBAL_GPU_INDEX = 0


def train_teacher_single_processes(args):
    """Single process of training a teacher
        Note: it will be called by a pool of processes
        Note: it will train teacher on GPU (i % GPU_COUNT)

    Args:
        args (tuple): 
            (i, prms, partitions, x_test, y_test, train_teacher_fn, GPU_COUNT)
            i (_type_): index of the teacher to train
            GPU_COUNT
    """

    (i, prms, partitions, x_test, y_test, train_teacher_fn, GPU_COUNT) = args
    GPU_ID = i % GPU_COUNT

    print(f"Training teacher {i} on GPU {GPU_ID}")
    logger.info(f"Training teacher {i} on GPU {GPU_ID}")
    with torch.cuda.device(GPU_ID):
        teacher_folder = prms.teachers_dir / f'teacher_{i}'
        if teacher_folder.is_dir():
            logger.info(
                f"Teacher {i} of {prms.pate.n_teachers} has already been trained."
            )
            return

        x_train = partitions[i][0]
        y_train = partitions[i][1]
        teacher, statistics = train_teacher_fn(
            model_prms=prms.models,
            data_name=prms.data.data_name,
            train_data=(x_train, y_train),
            test_data=(x_test, y_test),
            seed=prms.pate.seed,
        )

        # save teacher and statistics together with parameters
        teacher_dir = prms.teachers_dir / f'teacher_{i}'
        makedirs(teacher_dir, exist_ok=True)
        teacher.save(path=teacher_dir)
        stats_path = prms.resources.out_dir / 'stats_teachers.csv'
        statistics.update(prms.pate.__dict__)
        pd.DataFrame(data=[statistics.values()],
                     columns=statistics.keys()).to_csv(
                         path_or_buf=stats_path,
                         mode='a',
                         header=not stats_path.is_file())


def main(prms: ExperimentParameters,
         data_factory: DataFactory,
         n_reduce_teachers: int = 0,
         max_teachers=None,
         GPU_COUNT=4):
    """
    This method trains an ensemble of teacher models as part of the PATE pipeline.

    It randomly allocates subsets of the dataset to teachers depending on the privacy personalization.
    The allocation is stored and then the teachers are trained and stored.

    @param prms: Parameters for the experiment, used for training the complete ensemble.
    @param data_factory: DataFactory object to retrieve data splits.
    @param n_reduce_teachers: Reduce number of trained teachers by n
                              without changing partitioning of data.
                              Used for hyperparameter search for teachers.
    """

    x_test, y_test = data_factory.data_test(seed=prms.pate.seed)

    partitions, _, _ = prepare_private_partitions(
        prms=prms, data_private=data_factory.data_private(seed=prms.pate.seed))

    # train the teacher ensemble
    train_teacher_fn = ExperimentFactory(prms.data.data_name).step_teachers
    teacher_count = prms.pate.n_teachers - n_reduce_teachers
    if max_teachers is not None:
        teacher_count = min(teacher_count, max_teachers)

    def teacher_args_generator():
        for i in range(teacher_count):
            yield (i, prms, partitions, x_test, y_test, train_teacher_fn,
                   GPU_COUNT)

    #teacher_args = [(i, prms, partitions, x_test, y_test, train_teacher_fn, GPU_COUNT) for i in range(teacher_count)]
    print("About to train teachers")

    process_count = 20
    # GPUs  = 4
    max_processes_per_gpu = 8

    teachers = list(range(teacher_count))
    # indices of teachers to train

    teacher_groups = [
        teachers[ind:ind + (process_count)]
        for ind in range(0, len(teachers), (process_count))
    ]
    start = datetime.now()

    for teacher_group in teacher_groups:
        now = datetime.now()
        print(f"parsing teachers: {teacher_group}")
        print("------")
        print(f"Time since beginning of training : {now - start}")
        teacher_args = [(i, prms, partitions, x_test, y_test, train_teacher_fn,
                         GPU_COUNT) for i in teacher_group]

        with Pool(processes=process_count) as pool:
            pool.map(train_teacher_single_processes, teacher_args)

        print(f"Time for teacher group training : {datetime.now() - now}")
        #pool.map(train_teacher_single_processes, teacher_args)
        #range(teacher_count))
