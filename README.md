# A Framework to Implement and Experiment with personalized PATE

Private Aggregation of Teacher Ensembles (PATE) provides the protection and
accounting of privacy within machine learning (see "Semi-Supervised Knowledge Transfer for Deep Learning from Private Training Data", and "Scalable Private Learning with PATE", both from Papernot et al.).

This repository provides an implementation of PATE, as well as extensions of it to enable the personalization of privacy preservation, and a framework for experimentation. The code is designed in an object-oriented manner.

## Development and Usage
This codebase uses `python3.9`. 
### Installation
This library is developed using [Poetry](https://python-poetry.org/), evidenced by the `pyproject.toml`. However, it can be installed either through Poetry or with `pip` + your favorite virtual environment.

#### Installation Using a Virtual Environment [Tested and Supported]
1. Create a virtual environment `python3.9 -m venv venv`
2. Source this environment `source venv/bin/activate`
3. From the base of the codebase, run `pip install -e .`   

#### Installation Using Poetry [Not supported by authors]
While `poetry` (version 1.2.1) is used to manage the dependencies, and the authors use poetry, tests are run using `venv` and so the authors only commit to supporting installation using `virtualenv` or `venv`

1. Install Poetry version 1.2.1 (`curl -sSL https://install.python-poetry.org | python3 -`)
2. Navigate to the base of the codebase.
3. Run `poetry shell`
4. Run `poetry install`



Both of these methods for installation installs scripts for executing `individualized-pate` with various datasets, and puts them on your `PYTHONPATH`. Running `poetry shell` or `source venv/bin/activate` will shell into the virtual environments with the code installed, and will allow you to run the executables directly.
### Testing
Simply run 
```bash
pytest tests/ -v
```

### Execution
- Create an experiment plan according to the data class `ExperimentPlan`
  or use the ones from `individualized_pate/individualized-pate/experiment_plans`.

- 
- To run a single Experiment plan, execute the following:
  ```bash
  ppp-run --params_path path/to/experiment_plan.yaml --data_dir path/to/data --out_dir path/to/output_dir
  ```
- To run a set of experiment plans, execute the following:
  ```bash
  ppp-run-experiment-set --set_params_dir path/to/dir --data_dir path/to/data --out_root_dir path/to/output_dir
  ```
  This assumes that the folder `path/to/data` contains the necessary files for the dataset specified in `experiment_plan.yaml`.

### Usage on Slurm Cluster
- On a cluster that manages resources with slurm it is necessary to create a "job-script" that wraps the scripts from `individualized-pate`. See example below:

Note: Do this after downloading the code, and installing the codebase in a virtualenvironment, that can be sourced from the slurm-script.
  ```bash
  #!/usr/bin/env bash

  #SBATCH --job-name=study_mnist
  #SBATCH --mail-user=mail.user@domain.com
  #SBATCH --mail-type=end
  #SBATCH --output=study_mnist_%j.out

  #SBATCH --qos=standard
  #SBATCH --time=02:00:00
  #SBATCH --ntasks=1
  #SBATCH --mem 2G
  #SBATCH --partition=gpu
  #SBATCH --gres=gpu:1

  module load Python/3.9.5-GCCcore-10.3.0
  module load CUDA/11.3.1
  module load cuDNN/8.2.1.32-CUDA-11.3.1
  source /path/to/venv/bin/activate # source the virtualenvironment that had previously been set up.
  ppp-run -c $1 -d $2 -o $3
  ```
- The above job-script wraps the pate_mnist script and is executed with sbatch:
  ```bash
  sbatch job_script.bash path/to/experiment_plan.yaml path/to/data path/to/output_dir
  ```

### Creating a new ExperimentPlan
- Some experiment plans are available under `experiment_plans`.
- To create a new experiment plan, it is advised to start from one of these as a template.
- The documentation found under `individualized-pate/individualized_pate/studies/parameters.py` gives specific details on the different parameters.

## Architecture
**TODO:** Nomenclature and folder structure needs to be fixed.

## Code Structure
```
├── assets
│   └─ ...
├── CHANGELOG.md
├── experiment_plans
│   ├── paper_experiments
│   │   ├── <Experiment plan yamls and scripts>
├── individualized_pate
│   ├── data
│   │   └── <Code for the generation and management of data>
│   ├── experiments
│   │   ├── Libraries for wrappers for the Individualize-PATE experiments
│   │   └── pytorch # 
│   │       └── libraries for training teachers and students in pytorch
│   ├── hprms_search.py
│   ├── __init__.py
│   ├── main.py
│   ├── models
│   │   ├── classifiers.py
│   │   ├── __init__.py
│   │   └── pytorch
│   │       └── Libraries for training models with different architectures
│   ├── privacy
│   │   └── General privacy accounting and management, in general and for PATE 
│   ├── studies
│   │   ├── Code for the running of training multiple students and teachers (calls `experiments.pytorch`)
│   │   ├── train_students.py
│   └── └── train_teachers.py
├── notebooks
│   └── <Notebooks for the analysis and plot generation of MNIST and Adult results> 
│       
├── poetry.lock
├── pyproject.toml
├── README.md
└── tests
    └── <tests>
```

### Experiments, Plans and Sets
- An **ExperimentPlan** specifies multiple of Experiments over multiple seeds,
  budgets etc. This does not include other parameters as used datasets.
  A parameter file always specifies one experiment plan.
- **Experiment**s are derived from the experiment plan during runtime and executed sequentially.
- **ExperimentSets** are defined by a set of multiple parameter files. They can be defined over
  multiple datasets, model archetectures and other parameters that are not unfolded from experiment plans to single experiments during runtime.

### Structure
- **data** code to load and handle datasets.
- **experiments** contains substeps of the PATE framework (named experiments for historical reasons).
- **models** contains definitions of specific machine learning models.
- **privacy** is the core of this package. It contains the definitions of all
  PATE variants, and the computations for privacy accounting.
- **studies**
  Contains code for executing the higher level steps (teacher training, voting, student training) of the pate framework
  that in turn call functions from experiments to execute substeps.

### PATE Implementation
- PATE is implemented as a class with many attributes.
- The fundamental attribute of a PATE object is its aggregator which contains
  a collector.
- The kind of **aggregator** determines which labels are taken or rejected.
- The kind of **collector** determines which of the teacher votes are collected and
  how. It also specifies the (personalized) accounting of privacy expenditure.
- Currently, three kinds of aggregators (default, confident, interactive), and
  four kinds of collectors (GNMax, uGNMax, vGNMax, wGNMax) are defined.
  **TODO:** Interactive aggregator does not seem available.
- The data-dependent (tight) privacy bound was taken from
  https://github.com/tensorflow/privacy/blob/master/research/pate_2018/core.py.


### PATE Pipeline
- **Data Selection**:
  - The dataset of interest is partitioned into **private**, **public**, and
    **test** parts which are stored in the output folder.
  - The private dataset is randomly divided further and given to the teachers
    according to the privacy personalization.
  - New nested folders are created named
    `teachers/<n_teachers>_teachers_<budgets>_<distribution>/seed_<seed>/
    <collector>`. Therein, the concrete allocation and the concrete budgets of
    all private data are stored via pickle files named `mapping.pickle` and
    `budgets.pickle`.
- **Teachers' Training**:
  - By reading the pickle files, all subsets of the private data are given to
    their corresponding teachers.
  - The trained teachers are stored via pickle files in the above named folder
    each in a newly created folder named `teacher_<teacher_num>`.
  - The results of the teacher training are stored in the output folder under `stats_teachers.csv`.
- **Voting**:
  - For the voting, all teachers are loaded subsequently and predict labels for
    all public data. These labels are aggregated depending on the selected
    aggregator.
  - It produces labels until any privacy budget is exhausted or until a
    specified number of labels is acquired.
  - Labels are produced for a random shuffle of the public dataset.
  - New nested folders are created named `voting/
    <n_teachers>_teachers_<budgets>_<distribution>/seed_<seed>_<seed2>` within
    output folder folder. Therein, the
    produced labels are stored together with the corresponding features and the
    true labels.
  - The result of each voting process is stored as one line of the file
    `stats_votings.csv` in the output directory.
- **Student's Training**:
  - For the student's training, the data that was stored after the voting
    process are loaded. The number of used data is either determined by an
    integer, or by privacy budgets so that only labels are used until the
    budget was exhausted in the voting.
  - The training set is divided into each a part for validation and a part for
    training with ratios specified in the used classifier.
  - Depending on the kind of data, a specific data augmentation can be used.
  - The result of each student's training is stored as one line of the file
    `stats_students.csv` that is created in the output folder.

## Personalized Variants
There are three different personalized PATE variants implemented as collectors
in this package, namely *upsampling*, *vanishing*, and *weighting* GNMax
('uGNMax', 'vGNMax', and 'wGNMax').

- **uGNMax** regards the number of teachers that where trained on the same data
  point to account its corresponding privacy cost.
- **vGNMax** randomly determines which teacher's prediction is used in every
  voting corresponding to its data points' privacy budgets. Accordingly,
  an aggregator that uses the vGNMax collector calculates the privacy costs of
  all data corresponding to each teacher.
- **wGNMax** determines the weights of all teachers corresponding to their data
  points' privacy budgets. Each teacher's prediction is weighted so that it
  influences the voting and has privacy costs according to its weight.

Before the teachers are trained, data points are allocated to the teachers
according to their privacy budgets, s.t. points with equal or at least similar
budgets have the same teacher. In the case of uGNMax, each private data point
is duplicated according to its budget and randomly given to different teachers.

## Extension and cleanup of framework

### Tighter privacy bounds
* The PATE pipeline uses a method to convert from RDP to (epsilon, delta)-DP as presented in [Papernot et al.](2018http://arxiv.org/abs/1802.08908).
The improved method implemented [here](https://github.com/cleverhans-lab/capc-iclr/blob/fe9d3530929ed4a13cbf6cf0c1d35cf55dfc8de3/learning/analysis/pate.py#L64) should give considerably better results.

### Adding Datasets
* For adding a dataset, a method to load said dataset should be implemented under `individualized-pate/individualized_pate/data/load_datasets.py`.
* Subsequently, `data_augmentation.py` and `data_factory.py` in the same folder should be extended.

### Adding models
* Add code for the models under `individualized-pate/individualized_pate/models`.
* Make the model inherit the `Classifier` class (or have some wrapper).
* The substeps of the PATE pipeline steps (found in `individualized-pate/individualized_patelized_patelized_patelized_pate/experiments`) partially hard-code which model to use. The `architecture` parameter is passed to select from implementations.

### Reorganizing folder structure:
* What is denoted by one Experiment grew organically and should be fixed. As is, an ExperimentPlan specifies multiple Experiments which each is one execution of the PATE pipeline. The folder `individualized-pate/individualized_pate/experiments` instead contains code that specifies substeps executed during the steps teacher training, voting and student training.
* Executing an ExperimentPlan was formally called running a study, therefore the historical name of `individualized-pate/individualized_pate/studies`. This should be renamed.

### Reducing complexity
* One of the main problems of the current framework is high complexity in terms of dependencies of the single components as well as usage. Below are several proposed steps to reduce this compexity.
* Simplify ExperimentPlan
  * One ExperimentPlan allows to specify different aggregators. This should be removed for simplicity.
  * The ExperimentPlan allows to specify different teacher numbers, noise scales and other PATE parameters. This should be removed for simplicity.
* Clarify the structure of higher level pate steps (found in `individualized-pate/individualized_pate/study`) and the respective substeps (found in `individualized-pate/individualized_pate/experiments`).
* The `ExperimentPlan` (found in `individualized-pate/individualized_pate/study/parameters.py`) class was introduced to have one location for all parameters and derived properties. This could be followed up by unifying logging, still spread over multiple locations (higher level steps of PATE pipeline and the respective substeps).
* Code for selecting and training a model is currently distributed over multiple locations:
  * `architecture` parameter.
  * The substeps of the PATE pipeline `individualized-pate/individualized_pate/experiments`
  * The `Classifier` class again specifies train, eval etc. methods for the specific models.
  * These many levels of abstraction come with high complexity that could be reduced.