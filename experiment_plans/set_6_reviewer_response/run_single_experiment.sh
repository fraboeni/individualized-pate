#!/bin/bash
set -x
# Note: run script in a tmux terminal

# Arg 1 - YAML name 
# Arg 2 - YAML dir [ optional ]
# Arg 3 - time to sleep [ optional ]


### Params that don't changes:
BASE_OUT_DIR=$HOME/code/data/out/
datadrive_BASE_OUT_DIR=/datadrive1/CleverHans/individualized_pate
CODE_DIR=$HOME/code/personalized-pate/per-point-pate
INPUT_DATA_DIR=$HOME/code/data/inputs/
YAML_DIR=$CODE_DIR/experiment_plans/set_6_reviewer_response/

### ARGs
EXPERIMENT_TYPE=${1:-"mnist_upsample_figure_2"}
YAML_DIR=${2:-$YAML_DIR}
TIME_TO_SLEEP=${3:-0} # 7200 is 2 hours, in seconds
OUT_SUFFIX=${4:-}
OUT_DIR=$BASE_OUT_DIR/${EXPERIMENT_TYPE}__${OUT_SUFFIX}
mkdir -p $OUT_DIR    

echo $EXPERIMENT_TYPE
echo "Experiment dir : $YAML_DIR/$EXPERIMENT_TYPE.yaml" 


EXECUTABLE="ppp-run -p $YAML_DIR/$EXPERIMENT_TYPE.yaml -o $OUT_DIR -d $INPUT_DATA_DIR"


## logging
echo "Will sleep for $TIME_TO_SLEEP"
echo "Will run command : \'$EXECUTABLE\'"
sleep $TIME_TO_SLEEP

## Run the experiment
pushd $CODE_DIR
#poetry shell
popd

#poetry run 
ppp-run -p $YAML_DIR/$EXPERIMENT_TYPE.yaml -o $OUT_DIR -d $INPUT_DATA_DIR

