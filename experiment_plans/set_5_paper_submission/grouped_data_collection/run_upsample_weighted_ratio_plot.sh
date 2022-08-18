#!/bin/bash
set -x

# run a script in a tmux terminal
BASE_OUT_DIR=$HOME/code/data/out/
CODE_DIR=$HOME/code/personalized-pate/per-point-pate

INPUT_DATA_DIR=$HOME/code/data/inputs/
YAML_DIR=$CODE_DIR/experiment_plans/set_5_paper_submission/


for i in 1 2 3 4 5 6 7 8 9;
do 
    echo $i
    # run the run_single_mnist_script for each of the yamls for upsample
    $HOME/code/personalized-pate/per-point-pate/experiment_plans/set_5_paper_submission/run_single_mnist_script.sh upsample_weighted_ratio_plot_$i  $HOME/code/personalized-pate/per-point-pate/experiment_plans/set_5_paper_submission/upsample_weighted_ratio_plot/

done
