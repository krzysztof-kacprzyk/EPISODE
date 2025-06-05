# Skip the Equations: Learning Behavior of Personalized Dynamical Systems Directly From Data

This is the official repository for the paper "No Equations Needed: Learning System Dynamics Without Relying On Closed-Form ODEs"

## Dependencies
You can install all required dependencies using conda and the following command
```
conda env create -n episode --file environment.yml
```
This will also install `episode` (the main module) in editable mode.

## Running all experiments
To run all experiments navigate to `experiments` using
```
cd experiments
``` 
and run
```
./run_scripts/main_experiemnts.sh
./run_scripts/tumor_example.sh
./run_scripts/tacrolimus_study.sh
./run_scripts/sample_trajectories.sh
```

Or you can simply run 
```
./run_scripts/run_all.sh
```

The results will be saved in
```
experiments/results/
```

## Figures and tables
Jupyter notebooks used to create all figures and tables in the paper can be found in `experiments/analysis`.

## Citation

Kacprzyk, K., Piskorz, J., & van der Schaar, M. (2025). Skip the Equations: Learning Behavior of Personalized Dynamical Systems Directly From Data. *Forty-second International Conference on Machine Learning*.

```
@inproceedings{Kacprzyk.SkipEquationsLearning.2025,
  title = {Skip the {{Equations}}: {{Learning Behavior}} of {{Personalized Dynamical Systems Directly From Data}}},
  booktitle = {Forty-Second {{International Conference}} on {{Machine Learning}}},
  author = {Kacprzyk, Krzysztof and Piskorz, Julianna and {van der Schaar}, Mihaela},
  year = {2025}
}
```




