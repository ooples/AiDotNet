---
title: "ProfileReport"
description: "A comprehensive profiling report containing all collected metrics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diagnostics`

A comprehensive profiling report containing all collected metrics.

## Properties

| Property | Summary |
|:-----|:--------|
| `Entries` | Gets the profiled entries. |
| `StartTime` | Gets the start time of the profiling session. |
| `Stats` | Gets a dictionary of operation name to statistics. |
| `Tags` | Gets custom tags associated with this report. |
| `TotalOperations` | Gets the total number of operations tracked. |
| `TotalRuntime` | Gets the total runtime of the profiling session. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CompareTo(ProfileReport,Double)` | Compares this report to a baseline and identifies regressions. |
| `GetAllStats` | Gets statistics for all operations, sorted by total time descending. |
| `GetHotspots(Int32)` | Gets the top N operations ordered by total time (hotspots). |
| `GetSlowOperations(Double)` | Gets operations that exceed the specified P95 threshold. |
| `GetStats(String)` | Gets statistics for a specific operation. |
| `GetTopOperations(Int32)` | Gets the top N operations by total time. |
| `ToCsv` | Exports the report to CSV format. |
| `ToDictionary` | Exports the report to a dictionary for serialization. |
| `ToJson` | Exports the report to JSON format. |
| `ToMarkdown` | Exports the report to Markdown format. |

