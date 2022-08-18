#!/bin/bash
set -x

# run a script in a tmux terminal
BASE_OUT_DIR=$HOME/code/data/out/
CODE_DIR=$HOME/code/personalized-pate/per-point-pate

INPUT_DATA_DIR=$HOME/code/data/inputs/
YAML_DIR=$CODE_DIR/experiment_plans/set_6_reviewer_response/svhn_table_1
TIME_TO_SLEEP=0

for i in  2 3 4
do 
    for EXP in "weight" "upsample"  "vanish" "pate"
    do
    # run the run_single_experiment for each of the yamls for upsample
        YAML_NAME="svhn_${EXP}_row_1"
        SUFFIX_NAME="exp_${i}"
        
        /$CODE_DIR/experiment_plans/set_6_reviewer_response/run_single_experiment.sh $YAML_NAME $YAML_DIR $TIME_TO_SLEEP $SUFFIX_NAME

    done
done
