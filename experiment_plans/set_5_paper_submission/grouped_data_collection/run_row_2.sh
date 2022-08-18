#!/bin/bash
set -x

# run a script in a tmux terminal
BASE_OUT_DIR=$HOME/code/data/out/
CODE_DIR=$HOME/code/personalized-pate/per-point-pate

INPUT_DATA_DIR=$HOME/code/data/inputs/
YAML_DIR=$CODE_DIR/experiment_plans/set_5_paper_submission/

for EXP in "pate" "upsample" "vanish"
do
 # run the run_single_mnist_script for each of the yamls for upsample
$HOME/code/personalized-pate/per-point-pate/experiment_plans/set_5_paper_submission/run_single_mnist_script.sh mnist_${EXP}_row_2__2

done
