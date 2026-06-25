---
title: "StageCallbacks<T, TInput, TOutput>"
description: "Callbacks for training stage events."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Options`

Callbacks for training stage events.

## Properties

| Property | Summary |
|:-----|:--------|
| `OnBatchComplete` | Called after each batch within the stage. |
| `OnEpochComplete` | Called after each epoch within the stage. |
| `OnStageComplete` | Called when the stage completes successfully. |
| `OnStageError` | Called if the stage encounters an error. |
| `OnStageStart` | Called before the stage starts. |

