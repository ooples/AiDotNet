---
title: "TrainingMonitor<T>"
description: "Implementation of training monitoring system for tracking model training progress."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TrainingMonitoring`

Implementation of training monitoring system for tracking model training progress.

## How It Works

**For Beginners:** This is a complete implementation of a training monitor that tracks
all aspects of your model training in real-time.

Features include:

- Real-time metric logging (loss, accuracy, etc.)
- Resource usage tracking (CPU, memory, GPU)
- Progress estimation and ETA calculation
- Automatic issue detection (NaN values, stalled training, etc.)
- Export to JSON/CSV formats

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TrainingMonitor` | Initializes a new instance of the TrainingMonitor class. |

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

