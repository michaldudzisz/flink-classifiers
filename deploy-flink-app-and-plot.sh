#!/bin/bash

#source ./deploy-flink-app.sh

export PYTHON_VENV_NAME=mgr-venv
source "${PYTHON_VENV_NAME}/bin/activate"

export RESULTS_DIRECTORY="/home/michal/Documents/mgr/flink-classifiers/results"
export EXPERIMENT_ID=3cda8d69-320b-48ca-a211-5bab024a84fe

export PLOTS_DIR="$RESULTS_DIRECTORY/${EXPERIMENT_ID}/plots"
export DESCRIPTION="dsc"
python3 misc/plotter/results_plotter.py --description="$DESCRIPTION" --plotsDir="$PLOTS_DIR"


#python3 misc/plotter/new_results_plotter.py --description="new" --plotsDir="$PLOTS_DIR"


#if [ -z "$DESCRIPTION" ]; then
#  python3 misc/plotter/results_plotter.py
#else
#  python3 misc/plotter/results_plotter.py --description="$DESCRIPTION"
#fi

#rm -rf "${RESULTS_DIRECTORY}/${EXPERIMENT_ID}"
