---
title: "TrainingMonitorBase<T>"
description: "Base class for training monitoring implementations."
section: "API Reference"
---

`Base Classes` · `AiDotNet.TrainingMonitoring`

Base class for training monitoring implementations.

## How It Works

**For Beginners:** This abstract base class provides common functionality for training
monitoring systems. It handles session management, metric storage, and provides helper methods
for tracking training progress while leaving specific visualization to derived classes.

Key features:

- Thread-safe session and metric management
- Common metric aggregation utilities
- Resource usage tracking support
- Progress estimation helpers

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TrainingMonitorBase` | Initializes a new instance of the TrainingMonitorBase class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CheckForIssues(String)` | Checks for potential training issues and returns warnings. |
| `CreateVisualization(String,List<String>,String)` | Creates a visualization of training metrics. |
| `EndSession(String)` | Ends the current monitoring session. |
| `ExportData(String,String,String)` | Exports monitoring data to a file. |
| `GenerateSessionId` | Generates a unique session ID. |
| `GetCurrentMetrics(String)` | Gets the current metrics for a session. |
| `GetMetricHistory(String,String)` | Gets the history of a specific metric. |
| `GetResourceUsage(String)` | Gets the current resource usage. |
| `GetSession(String)` | Gets a session by ID. |
| `GetSpeedStats(String)` | Gets statistics about training speed. |
| `LogMessage(String,LogLevel,String)` | Logs a text message or event during training. |
| `LogMetric(String,String,,Int32,Nullable<DateTime>)` | Records a metric value for the current training step. |
| `LogMetrics(String,Dictionary<String,>,Int32)` | Records multiple metrics at once. |
| `LogResourceUsage(String,Double,Double,Nullable<Double>,Nullable<Double>)` | Records system resource usage. |
| `OnEpochEnd(String,Int32,Dictionary<String,>,TimeSpan)` | Records the end of a training epoch with summary metrics. |
| `OnEpochStart(String,Int32)` | Records the start of a new training epoch. |
| `SerializeToJson(Object)` | Serializes an object to JSON. |
| `StartSession(String,Dictionary<String,Object>)` | Starts monitoring a training session. |
| `UpdateProgress(String,Int32,Int32,Int32,Int32)` | Updates the training progress information. |
| `ValidateExportPath(String)` | Validates and sanitizes a file path for export. |
| `ValidateSessionExists(String)` | Validates that a session exists. |

## Fields

| Field | Summary |
|:-----|:--------|
| `JsonSettings` | JSON serialization settings for consistent serialization. |
| `Sessions` | Active monitoring sessions keyed by session ID. |
| `SyncLock` | Lock object for thread-safe operations. |

