#!/bin/bash
set -x

# run a script in a tmux terminal
BASE_OUT_DIR=$HOME/code/data/out/
CODE_DIR=$HOME/code/personalized-pate/per-point-pate

INPUT_DATA_DIR=$HOME/code/data/inputs/
YAML_DIR=$CODE_DIR/experiment_plans/set_6_reviewer_response/mnist_table_1

for EXP in "pate" "upsample" "weight" "vanish" 
do
 # run the run_single_mnist_script for each of the yamls for upsample
    YAML_NAME="mnist_${EXP}_row_1"

    /$CODE_DIR/experiment_plans/set_6_reviewer_response/run_single_experiment.sh $YAML_NAME $YAML_DIR

    YAML_NAME="mnist_${EXP}_row_2"

    /$CODE_DIR/experiment_plans/set_6_reviewer_response/run_single_experiment.sh $YAML_NAME $YAML_DIR
done
