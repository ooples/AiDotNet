---
title: "IBatchSampler"
description: "Extended interface for samplers that support batch-level sampling."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Extended interface for samplers that support batch-level sampling.

## How It Works

Some samplers need to operate at the batch level rather than sample level,
for example to ensure each batch contains samples from all classes (stratified batching).

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for batch-level sampling. |
| `DropLast` | Gets or sets whether to drop the last incomplete batch. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetBatchIndices` | Returns an enumerable of index arrays, where each array represents one batch. |

