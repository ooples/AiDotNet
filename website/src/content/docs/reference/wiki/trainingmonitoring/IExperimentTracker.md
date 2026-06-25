---
title: "IExperimentTracker"
description: "Interface for experiment tracking systems."
section: "API Reference"
---

`Interfaces` · `AiDotNet.TrainingMonitoring.ExperimentTracking`

Interface for experiment tracking systems.

## How It Works

**For Beginners:** Experiment tracking is essential for machine learning
workflows. It helps you:

- Keep track of different experiments and their parameters
- Compare results across runs
- Reproduce experiments by logging all relevant information
- Organize and search through past experiments

Think of it like MLflow or Weights & Biases - it's a central place
to log everything about your training runs.

Key concepts:

- Experiment: A named project (e.g., "image-classification")
- Run: A single execution of training (e.g., "run_20241220_143052")
- Parameters: Configuration values (learning_rate, batch_size, etc.)
- Metrics: Measured values (loss, accuracy, etc.)
- Artifacts: Files produced (model weights, plots, etc.)
- Tags: Labels for organization (dev, production, etc.)

## Properties

| Property | Summary |
|:-----|:--------|
| `ActiveExperiment` | Gets the active experiment name, if any. |
| `ActiveRunId` | Gets the active run ID, if any. |
| `TrackingUri` | Gets the tracking URI. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CompareRuns(String[])` | Compares multiple runs. |
| `CreateExperiment(String,String,Dictionary<String,String>)` | Creates or gets an experiment by name. |
| `DeleteExperiment(String)` | Deletes an experiment and all its runs. |
| `DeleteRun(String)` | Deletes a run. |
| `EndRun(RunStatus)` | Ends the active run. |
| `GetExperiment(String)` | Gets an experiment by name. |
| `GetMetricHistory(String,String)` | Gets metric history for a run. |
| `GetRun(String)` | Gets a run by ID. |
| `ListExperiments` | Lists all experiments. |
| `ListRuns(String,String,String,Int32)` | Lists runs in an experiment. |
| `LogArtifact(String,String)` | Logs an artifact file. |
| `LogArtifacts(String,String)` | Logs all files in a directory as artifacts. |
| `LogMetric(String,Double,Nullable<Int64>)` | Logs a metric at a specific step. |
| `LogMetrics(Dictionary<String,Double>,Nullable<Int64>)` | Logs multiple metrics at a specific step. |
| `LogModel(String,String,Dictionary<String,Object>)` | Logs a model artifact. |
| `LogParameter(String,String)` | Logs a parameter (configuration value). |
| `LogParameters(Dictionary<String,String>)` | Logs multiple parameters. |
| `RestoreRun(String)` | Restores a deleted run. |
| `SearchRuns(IEnumerable<String>,String,String,Int32)` | Searches for runs matching criteria. |
| `SetExperiment(String)` | Sets the active experiment for subsequent runs. |
| `SetTag(String,String)` | Sets a tag on the active run. |
| `SetTags(Dictionary<String,String>)` | Sets multiple tags on the active run. |
| `StartRun(String,Dictionary<String,String>,String)` | Starts a new run within the active experiment. |

