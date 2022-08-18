#!/bin/bash
set -x

# run a script in a tmux terminal
BASE_OUT_DIR=$HOME/code/data/out/
CODE_DIR=$HOME/code/personalized-pate/per-point-pate

INPUT_DATA_DIR=$HOME/code/data/inputs/
YAML_DIR=$CODE_DIR/experiment_plans/set_5_paper_submission/table_4_weighting/

## 25-75

for YAML_NAME  in "mnist_weight_75_25__ln16.yaml" "mnist_weight_75_25__ln4.yaml"
do
    echo $YAML_NAME
    NAME_no_ext="${YAML_NAME%.*}"
    # run the run_single_mnist_script for each of the yamls for upsample
    echo $NAME_no_ext
    
    $HOME/code/personalized-pate/per-point-pate/experiment_plans/set_5_paper_submission/run_single_mnist_script.sh $NAME_no_ext $YAML_DIR

done