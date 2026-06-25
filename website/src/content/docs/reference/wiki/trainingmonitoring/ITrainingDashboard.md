---
title: "ITrainingDashboard"
description: "Interface for training dashboards that visualize metrics and training progress."
section: "API Reference"
---

`Interfaces` · `AiDotNet.TrainingMonitoring.Dashboard`

Interface for training dashboards that visualize metrics and training progress.

## How It Works

**For Beginners:** A training dashboard provides visual feedback about
your model's training progress. Think of it like TensorBoard - it shows
you graphs of loss, accuracy, and other metrics over time.

This library provides multiple dashboard implementations:

- HtmlDashboard: Generates interactive HTML reports
- LiveDashboard: Provides real-time updates via embedded web server
- ConsoleDashboard: Text-based visualization for terminal environments

Example usage:

## Properties

| Property | Summary |
|:-----|:--------|
| `IsRunning` | Gets whether the dashboard is running. |
| `LogDirectory` | Gets the log directory. |
| `Name` | Gets the dashboard name. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clear` | Clears all logged data. |
| `ExportTensorBoardFormat(String)` | Exports data in TensorBoard-compatible format. |
| `Flush` | Flushes any buffered data to storage. |
| `GenerateReport(String)` | Generates an HTML report of all logged data. |
| `GetHistogramData` | Gets all logged histogram series. |
| `GetScalarData` | Gets all logged scalar series. |
| `LogConfusionMatrix(String,Int64,Int32[0:,0:],String[],Nullable<DateTime>)` | Logs a confusion matrix. |
| `LogHistogram(String,Int64,Double[],Nullable<DateTime>)` | Logs a histogram (distribution) of values. |
| `LogHyperparameters(Dictionary<String,Object>,Dictionary<String,Double>)` | Logs hyperparameters. |
| `LogImage(String,Int64,Byte[],Int32,Int32,Nullable<DateTime>)` | Logs an image. |
| `LogModelGraph(String)` | Logs the model graph/architecture. |
| `LogPRCurve(String,Int64,Double[],Int32[],Nullable<DateTime>)` | Logs a precision-recall curve. |
| `LogROCCurve(String,Int64,Double[],Int32[],Nullable<DateTime>)` | Logs an ROC curve. |
| `LogScalar(String,Int64,Double,Nullable<DateTime>)` | Logs a scalar metric. |
| `LogScalars(Dictionary<String,Double>,Int64,Nullable<DateTime>)` | Logs multiple scalars at once. |
| `LogText(String,Int64,String,Nullable<DateTime>)` | Logs text. |
| `Start` | Starts the dashboard. |
| `Stop` | Stops the dashboard. |

