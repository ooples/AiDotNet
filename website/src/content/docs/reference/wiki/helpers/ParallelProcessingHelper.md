---
title: "ParallelProcessingHelper"
description: "Helper class for executing multiple tasks in parallel to improve performance."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Helper class for executing multiple tasks in parallel to improve performance.

## How It Works

**For Beginners:** This class helps your AI models run faster by doing multiple calculations 
at the same time, similar to having multiple people working on different parts of a project
simultaneously instead of one person doing everything sequentially.

## Methods

| Method | Summary |
|:-----|:--------|
| `ProcessTasksInParallel(IEnumerable<Func<>>,Nullable<Int32>)` | Executes multiple functions in parallel with controlled concurrency. |
| `ProcessTasksInParallel(IEnumerable<Task<>>,Nullable<Int32>)` | Executes multiple pre-created tasks in parallel batches with controlled concurrency. |

