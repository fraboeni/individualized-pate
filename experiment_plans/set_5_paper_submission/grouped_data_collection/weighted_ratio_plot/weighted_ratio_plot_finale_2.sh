#!/bin/bash

set -x

#       vanish weight
# 25-75   
# 75-25


BASE_CONFIG=$HOME/code/personalized-pate/per-point-pate/experiment_plans/set_5_paper_submission/weighted_ratio_plot/todo/
#  "vanish_weighted_ratio_plot_25_75" 
for NAME in "vanish_weighted_ratio_plot_25_75" #"weight_weighted_ratio_plot_50_50" 
do
    
    CONFIG_DIR=$BASE_CONFIG/$NAME
    pushd $CONFIG_DIR
    for i in $(seq 3 7)
    do
        $HOME/code/personalized-pate/per-point-pate/experiment_plans/set_5_paper_submission/run_single_mnist_script.sh ${NAME}__$i $CONFIG_DIR
    done 
    popd

done
