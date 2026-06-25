---
title: "TelemetryCollector"
description: "Collects telemetry data for deployed models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Runtime`

Collects telemetry data for deployed models.

## For Beginners

TelemetryCollector provides AI safety functionality. Default values follow the original paper settings.

## Methods

| Method | Summary |
|:-----|:--------|
| `Clear` | Clears all telemetry data. |
| `GetEvents(Int32)` | Gets the most recent recorded events ordered by timestamp descending. |
| `GetStatistics(String,String)` | Gets statistics for a model. |
| `RecordError(String,String,Exception)` | Records an error. |
| `RecordEvent(String,Dictionary<String,Object>)` | Records a telemetry event. |
| `RecordInference(String,String,Int64,Boolean)` | Records an inference execution. |

