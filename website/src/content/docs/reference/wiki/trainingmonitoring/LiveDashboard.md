---
title: "LiveDashboard"
description: "Provides a real-time training dashboard via embedded web server."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TrainingMonitoring.Dashboard`

Provides a real-time training dashboard via embedded web server.

## How It Works

**For Beginners:** LiveDashboard is like running TensorBoard - it starts
a local web server that you can open in your browser to see real-time
training progress. The charts update automatically as training progresses.

Example usage:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LiveDashboard(String,String,Int32,Boolean)` | Creates a new live dashboard. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AllowExternalConnections` | Gets or sets whether to allow external connections. |
| `IsRunning` |  |
| `LogDirectory` |  |
| `Name` |  |
| `Port` | Gets the port the server is listening on. |
| `RefreshIntervalMs` | Gets or sets the refresh interval in milliseconds for auto-updating clients. |
| `Url` | Gets the URL to access the dashboard. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clear` |  |
| `Dispose` | Disposes the live dashboard. |
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

