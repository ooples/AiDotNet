---
title: "ConsoleDashboard"
description: "Console-based training dashboard with ASCII charts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TrainingMonitoring.Dashboard`

Console-based training dashboard with ASCII charts.

## How It Works

**For Beginners:** ConsoleDashboard provides real-time training visualization
directly in your terminal using ASCII art. This is useful when you don't have
access to a web browser or want a lightweight monitoring solution.

Features:

- ASCII line charts for loss/accuracy
- Progress bars for training progress
- Real-time metric display
- Colored output for different metric types

Example usage:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConsoleDashboard(String,String)` | Creates a new console dashboard. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ChartHeight` | Gets or sets the chart height in characters. |
| `ChartWidth` | Gets or sets the chart width in characters. |
| `ClearOnRender` | Gets or sets whether to clear the screen on each render. |
| `IsRunning` |  |
| `LogDirectory` |  |
| `MaxMetricsDisplay` | Gets or sets the maximum number of metrics to display. |
| `Name` |  |
| `RefreshIntervalMs` | Gets or sets the refresh interval in milliseconds. |
| `UseColors` | Gets or sets whether to use colors. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clear` |  |
| `Dispose` | Disposes the console dashboard. |
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

