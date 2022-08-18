#!/bin/bash
set -x

DIRNAME=$1

# run a script in a tmux terminal
BASE_OUT_DIR=$HOME/code/data/out/
CODE_DIR=$HOME/code/personalized-pate/per-point-pate

INPUT_DATA_DIR=$HOME/code/data/inputs/
BASE_YAML_DIR=$CODE_DIR/experiment_plans/set_5_paper_submission/weighted_ratio_plot/
## 25-75

YAML_DIR=$BASE_YAML_DIR/$DIRNAME

for YAML_NAME in `ls $YAML_DIR | grep ".yaml" `
do
    echo $YAML_NAME
    NAME_no_ext="${YAML_NAME%.*}"
    # run the run_single_mnist_script for each of the yamls for upsample
    echo $NAME_no_ext
    
    $HOME/code/personalized-pate/per-point-pate/experiment_plans/set_5_paper_submission/run_single_mnist_script.sh $NAME_no_ext $YAML_DIR

done
