#!/bin/bash

# List of scripts to run
SCRIPTS=(
    run_scripts/main_experiments.sh
    run_scripts/tumor_example.sh
    run_scripts/tacrolimus_study.sh
    run_scripts/sample_trajectories.sh
)

for script in "${SCRIPTS[@]}"; do
    echo "Running $script..."
    bash "$script"
    echo
done