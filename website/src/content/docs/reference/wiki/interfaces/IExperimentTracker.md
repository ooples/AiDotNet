---
title: "IExperimentTracker<T>"
description: "Defines the contract for experiment tracking systems that log machine learning experiments."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for experiment tracking systems that log machine learning experiments.

## How It Works

An experiment tracker records information about machine learning experiments including parameters,
metrics, and artifacts to enable reproducibility and comparison of different training runs.

**For Beginners:** Think of an experiment tracker as a lab notebook for machine learning.
Just like a scientist records their experimental conditions and results, an experiment tracker
logs all the details of your machine learning model training - what settings you used, how well
it performed, and what models you created.

Key capabilities include:

- Creating and managing experiments (groups of related training runs)
- Logging hyperparameters (settings used for training)
- Recording metrics (performance measurements over time)
- Storing artifacts (models, plots, data files)
- Comparing different training runs
- Reproducing previous experiments

Why experiment tracking matters:

- Helps you keep track of what you've tried
- Makes it easy to reproduce good results
- Enables comparison between different approaches
- Provides audit trail for model development

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateExperiment(String,String,Dictionary<String,String>)` | Creates a new experiment to organize related training runs. |
| `DeleteExperiment(String)` | Deletes an experiment and all its associated runs. |
| `DeleteRun(String)` | Deletes a specific run. |
| `GetExperiment(String)` | Gets an existing experiment by its ID. |
| `GetRun(String)` | Gets an existing run by its ID. |
| `ListExperiments(String)` | Lists all experiments, optionally filtered by criteria. |
| `ListRuns(String,String)` | Lists all runs in an experiment, optionally filtered by criteria. |
| `SearchRuns(String,Int32)` | Searches for runs across all experiments based on criteria. |
| `StartRun(String,String,Dictionary<String,String>)` | Starts a new training run within an experiment. |

