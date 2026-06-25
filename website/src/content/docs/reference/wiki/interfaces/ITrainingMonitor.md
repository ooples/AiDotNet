---
title: "ITrainingMonitor<T>"
description: "Defines the contract for training monitoring systems that track and visualize model training progress."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for training monitoring systems that track and visualize model training progress.

## How It Works

A training monitor provides real-time visibility into the training process, tracking metrics,
system resources, and training state to help identify issues and optimize performance.

**For Beginners:** Think of a training monitor as a dashboard for your model training.
Just like a car dashboard shows speed, fuel, and engine temperature, a training monitor shows:

- Training metrics (loss, accuracy)
- System resources (CPU, GPU, memory usage)
- Training speed (iterations per second)
- Progress and estimated time remaining

Why training monitoring matters:

- Catch problems early (model not learning, overfitting, resource issues)
- Understand training dynamics and patterns
- Optimize resource usage
- Track progress on long-running training jobs
- Enable remote monitoring of training

## Methods

| Method | Summary |
|:-----|:--------|
| `CheckForIssues(String)` | Checks for potential training issues and returns warnings. |
| `CreateVisualization(String,List<String>,String)` | Creates a visualization of training metrics. |
| `EndSession(String)` | Ends the current monitoring session. |
| `ExportData(String,String,String)` | Exports monitoring data to a file. |
| `GetCurrentMetrics(String)` | Gets the current metrics for a session. |
| `GetMetricHistory(String,String)` | Gets the history of a specific metric. |
| `GetResourceUsage(String)` | Gets the current resource usage. |
| `GetSpeedStats(String)` | Gets statistics about training speed. |
| `LogMessage(String,LogLevel,String)` | Logs a text message or event during training. |
| `LogMetric(String,String,,Int32,Nullable<DateTime>)` | Records a metric value for the current training step. |
| `LogMetrics(String,Dictionary<String,>,Int32)` | Records multiple metrics at once. |
| `LogResourceUsage(String,Double,Double,Nullable<Double>,Nullable<Double>)` | Records system resource usage. |
| `OnEpochEnd(String,Int32,Dictionary<String,>,TimeSpan)` | Records the end of a training epoch with summary metrics. |
| `OnEpochStart(String,Int32)` | Records the start of a new training epoch. |
| `StartSession(String,Dictionary<String,Object>)` | Starts monitoring a training session. |
| `UpdateProgress(String,Int32,Int32,Int32,Int32)` | Updates the training progress information. |

