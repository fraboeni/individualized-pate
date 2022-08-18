#!/bin/bash
set -x

# run a script in a tmux terminal
BASE_OUT_DIR=$HOME/code/data/out/
CODE_DIR=$HOME/code/personalized-pate/per-point-pate/per_point_pate
YAML_DIR=$CODE_DIR/experiment_plans/set_5_paper_submission/
INPUT_DATA_DIR=$HOME/code/data/inputs/

pushd $CODE_DIR
poetry shell

COUNT=0
for TEACHER_COUNT in 28	22
#88  59	44    140  100
#200 400 800 1600 3200 6400
do
    TEACHER_DIR=$CODE_DIR/teacher_accuracy_tests/$TEACHER_COUNT

    echo "Experiment dir : $TEACHER_DIR" 

    CUDA_VISIBLE_DEVICE_COUNT=$((COUNT%4))
    
    EXECUTABLE="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICE_COUNT python $CODE_DIR/teacher_accuracy_tests.py -p $TEACHER_DIR/teacher_accuracy_test_$TEACHER_COUNT.yaml -d $CODE_DIR/teacher_accuracy_tests -o $TEACHER_DIR -m 10"

    echo "running command : \'$EXECUTABLE\'"
    
    TMUX_SESSION_NAME=tmux_${TEACHER_COUNT}__$COUNT
    echo "Tmux session : $TMUX_SESSION_NAME"

    tmux new-session -d -s $TMUX_SESSION_NAME "$EXECUTABLE"

    COUNT=$((COUNT+1))
    echo "------------------"

done

