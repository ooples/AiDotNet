---
title: "MetricCollection<T>"
description: "A collection of metrics, providing easy access by name and category."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Results.Core`

A collection of metrics, providing easy access by name and category.

## For Beginners

This is like a dictionary of metrics. You can access any metric by name
(e.g., collection["Accuracy"]) or iterate through all metrics. It also groups metrics by
category (Classification, Regression, etc.) for organized reporting.

## How It Works

MetricCollection provides a convenient way to store, access, and iterate over evaluation metrics.
It supports indexing by name, filtering by category, and various output formats.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MetricCollection` | Initializes a new empty metric collection. |
| `MetricCollection(IEnumerable<MetricWithCI<>>)` | Initializes a metric collection from an existing dictionary. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Categories` | Gets all categories in the collection. |
| `Count` | Gets the number of metrics in the collection. |
| `Item(String)` | Gets a metric by name. |
| `Names` | Gets all metric names in the collection. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(MetricWithCI<>)` | Adds a metric to the collection. |
| `AddOrUpdate(MetricWithCI<>)` | Adds or updates a metric in the collection. |
| `AddRange(IEnumerable<MetricWithCI<>>)` | Adds multiple metrics to the collection. |
| `Contains(String)` | Checks if a metric exists in the collection. |
| `GetBest(MetricDirection)` | Gets the best metric by the specified direction. |
| `GetByCategory(String)` | Gets all metrics in a specific category. |
| `GetEnumerator` | Returns an enumerator that iterates through the metrics. |
| `GetValidMetrics` | Gets all valid metrics (non-NaN, non-infinite values). |
| `GetWorst(MetricDirection)` | Gets the worst metric by the specified direction. |
| `System#Collections#IEnumerable#GetEnumerator` | Returns an enumerator that iterates through the metrics. |
| `ToDictionary` | Converts the collection to a dictionary. |
| `ToDictionaryByCategory` | Converts the collection to a dictionary grouped by category. |
| `ToTableString(Int32,Boolean)` | Formats all metrics as a table string. |
| `TryGetMetric(String,MetricWithCI<>)` | Tries to get a metric by name. |
| `Where(Func<MetricWithCI<>,Boolean>)` | Filters metrics by a predicate. |

