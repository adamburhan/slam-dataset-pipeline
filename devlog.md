# Section A: Definition of "done" for Jan 12
- `rpe_labels.csv` for several datasets
- training scripts run baseline models
- figure generated (boxplots across folds)
- short lit review

# Section B: Minimal pipeline contract
- `Dataset.get_sequence()`
- `SLAMSystem.run() -> artifacts (trajectory + tracking status)`
- `metrics.compute_rpe(policy=mask|impute_*) -> csv`

# Section C: Tracking loss as experimental factor
- policies: mask | impute_last | impute_cv
- always log tracked / valid flags
- evaluate:
  - tracked-only slice
  - failure-adjacent slice

notes:
- groundtruth loading is implemented in the dataset class
- estimated trajectory loading is implemented in trajectories and a parameter that is specified is format