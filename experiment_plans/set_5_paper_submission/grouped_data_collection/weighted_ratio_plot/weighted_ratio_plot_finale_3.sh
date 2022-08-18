#!/bin/bash

set -x

#       vanish weight
# 25-75   
# 75-25


BASE_CONFIG=$HOME/code/personalized-pate/per-point-pate/experiment_plans/set_5_paper_submission/weighted_ratio_plot/todo/
# "weight_weighted_ratio_plot_25_75" 
for NAME in "weight_weighted_ratio_plot_75_25" 
do
    
    CONFIG_DIR=$BASE_CONFIG/$NAME
    pushd $CONFIG_DIR
    for i in $(seq 8 15)
    do
        $HOME/code/personalized-pate/per-point-pate/experiment_plans/set_5_paper_submission/run_single_mnist_script.sh ${NAME}__$i $CONFIG_DIR
    done 
    popd
done
