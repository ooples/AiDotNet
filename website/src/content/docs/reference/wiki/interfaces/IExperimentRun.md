---
title: "IExperimentRun<T>"
description: "Represents a single training run within an experiment."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Represents a single training run within an experiment.

## How It Works

**For Beginners:** A run represents one attempt at training a model with specific settings.
Each run records everything about that training session: what settings you used (parameters),
how well it performed (metrics), and what it created (artifacts like the trained model).

Think of a run like a single entry in a lab notebook - it captures all the details of one
specific experiment attempt.

## Properties

| Property | Summary |
|:-----|:--------|
| `EndTime` | Gets the timestamp when the run ended. |
| `ExperimentId` | Gets the experiment ID this run belongs to. |
| `RunId` | Gets the unique identifier for this run. |
| `RunName` | Gets or sets the name of this run. |
| `StartTime` | Gets the timestamp when the run was started. |
| `Status` | Gets the current status of the run (Running, Completed, Failed, etc.). |
| `Tags` | Gets the tags associated with this run. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddNote(String)` | Adds a note or comment to the run. |
| `Complete` | Marks the run as completed successfully. |
| `Fail(String)` | Marks the run as failed with an optional error message. |
| `GetArtifacts` | Gets all artifacts logged for this run. |
| `GetLatestMetric(String)` | Gets the latest value for a specific metric. |
| `GetMetrics` | Gets all metrics logged for this run. |
| `GetNotes` | Gets all notes added to this run. |
| `GetParameters` | Gets all parameters logged for this run. |
| `LogArtifact(String,String)` | Logs an artifact (file) associated with this run. |
| `LogArtifacts(String,String)` | Logs a directory of artifacts. |
| `LogMetric(String,,Int32,Nullable<DateTime>)` | Logs a metric value at a specific step/iteration. |
| `LogMetrics(Dictionary<String,>,Int32,Nullable<DateTime>)` | Logs multiple metrics at once for a specific step. |
| `LogModel(IModel<,,>,String)` | Logs a trained model as an artifact. |
| `LogParameter(String,Object)` | Logs a single parameter value for this run. |
| `LogParameters(Dictionary<String,Object>)` | Logs multiple parameters at once. |

