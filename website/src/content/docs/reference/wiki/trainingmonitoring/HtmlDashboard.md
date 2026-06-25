---
title: "HtmlDashboard"
description: "Generates interactive HTML dashboards for training visualization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TrainingMonitoring.Dashboard`

Generates interactive HTML dashboards for training visualization.

## How It Works

**For Beginners:** HtmlDashboard creates beautiful, interactive HTML reports
that show your training progress. Unlike TensorBoard which requires running a server,
these HTML files can be opened directly in any web browser.

Features:

- Interactive charts with zoom and pan (using Chart.js)
- Real-time loss and accuracy curves
- Confusion matrix heatmaps
- Histogram distributions
- ROC and PR curves
- Hyperparameter logging

Example usage:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HtmlDashboard(String,String)` | Creates a new HTML dashboard. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoSave` | Gets or sets whether to auto-save on each log operation. |
| `AutoSaveInterval` | Gets or sets the auto-save interval in number of logs. |
| `IsRunning` |  |
| `LogDirectory` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clear` |  |
| `Dispose` | Disposes the dashboard. |
| `ExportTensorBoardFormat(String)` |  |
| `Flush` |  |
| `GenerateReport(String)` |  |
| `GetHistogramData` |  |
| `GetScalarData` |  |
| `LogConfusionMatrix(String,Int64,Int32[0:,0:],String[],Nullable<DateTime>)` |  |
| `LogHistogram(String,Int64,Double[],Nullable<DateTime>)` |  |
| `LogHyperparameters(Dictionary<String,Object>,Dictionary<String,Double>)` |  |
| `LogImage(String,Int64,Byte[],Int32,Int32,Nullable<DateTime>)` |  |
| `LogModelGraph(String)` |  |
| `LogPRCurve(String,Int64,Double[],Int32[],Nullable<DateTime>)` |  |
| `LogROCCurve(String,Int64,Double[],Int32[],Nullable<DateTime>)` |  |
| `LogScalar(String,Int64,Double,Nullable<DateTime>)` |  |
| `LogScalars(Dictionary<String,Double>,Int64,Nullable<DateTime>)` |  |
| `LogText(String,Int64,String,Nullable<DateTime>)` |  |
| `Start` |  |
| `Stop` |  |

