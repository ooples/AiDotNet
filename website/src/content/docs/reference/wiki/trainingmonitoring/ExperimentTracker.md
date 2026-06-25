---
title: "ExperimentTracker"
description: "Local file-based experiment tracker providing MLflow-compatible functionality."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TrainingMonitoring.ExperimentTracking`

Local file-based experiment tracker providing MLflow-compatible functionality.

## How It Works

**For Beginners:** ExperimentTracker stores all your experiment data locally
in a structured format. It's like having your own MLflow server without needing
to set up any infrastructure.

Directory structure:

Example usage:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExperimentTracker(String)` | Creates a new experiment tracker. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActiveExperiment` |  |
| `ActiveRunId` |  |
| `GitPath` | Gets or sets the path to the git executable. |
| `TrackingUri` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CompareRuns(String[])` |  |
| `CreateExperiment(String,String,Dictionary<String,String>)` |  |
| `DeleteExperiment(String)` |  |
| `DeleteRun(String)` |  |
| `Dispose` | Disposes the tracker. |
| `EndRun(RunStatus)` |  |
| `GetDefaultGitPath` | Gets the default path for git based on the current platform. |
| `GetExperiment(String)` |  |
| `GetMetricHistory(String,String)` |  |
| `GetRun(String)` |  |
| `ListExperiments` |  |
| `ListRuns(String,String,String,Int32)` |  |
| `LogArtifact(String,String)` |  |
| `LogArtifacts(String,String)` |  |
| `LogMetric(String,Double,Nullable<Int64>)` |  |
| `LogMetrics(Dictionary<String,Double>,Nullable<Int64>)` |  |
| `LogModel(String,String,Dictionary<String,Object>)` |  |
| `LogParameter(String,String)` |  |
| `LogParameters(Dictionary<String,String>)` |  |
| `RestoreRun(String)` |  |
| `SearchRuns(IEnumerable<String>,String,String,Int32)` |  |
| `SetExperiment(String)` |  |
| `SetTag(String,String)` |  |
| `SetTags(Dictionary<String,String>)` |  |
| `StartRun(String,Dictionary<String,String>,String)` |  |

