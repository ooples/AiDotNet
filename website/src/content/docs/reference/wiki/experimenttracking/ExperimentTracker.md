---
title: "ExperimentTracker<T>"
description: "Implementation of experiment tracking system for managing ML experiments and runs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ExperimentTracking`

Implementation of experiment tracking system for managing ML experiments and runs.

## How It Works

**For Beginners:** This is a complete implementation of MLflow-like experiment tracking.
It helps you organize and track all your machine learning experiments in one place.

Features include:

- Creating and managing experiments
- Starting and tracking training runs
- Logging parameters, metrics, and artifacts
- Searching and comparing runs
- Persistent storage of all experiment data

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExperimentTracker(String)` | Initializes a new instance of the ExperimentTracker class. |

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

