---
title: "BenchmarkFailurePolicy"
description: "Controls how the benchmark runner should behave when one or more suites fail."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Controls how the benchmark runner should behave when one or more suites fail.

## Fields

| Field | Summary |
|:-----|:--------|
| `ContinueAndAttachReport` | Run all requested suites and return a report even if some fail. |
| `ContinueAndThrowAggregate` | Run all requested suites, then throw an aggregate exception if any failed. |
| `FailFast` | Stop at the first failure and throw immediately. |

