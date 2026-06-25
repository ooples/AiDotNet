---
title: "WeightStreamingConfig"
description: "Configuration for weight streaming (paging large model weights to disk when they don't fit in RAM)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Deployment.Configuration`

Configuration for weight streaming (paging large model weights to disk
when they don't fit in RAM). Issue #1222 / weight-streaming v1.

## For Beginners

Modern foundation models (GPT-class LLMs,
vision-language models like PaLM-E) have so many parameters that the
raw weights don't fit in your machine's RAM — a 562B-parameter model
is ~2.25 TB at fp32. Weight streaming pages those weights to a fast
local disk and only loads the slice the model needs RIGHT NOW into
RAM, so the model can run on a laptop instead of a multi-GPU
workstation. The trade-off is throughput: cold disk reads add
latency. AiDotNet's defaults handle this for you (auto-enabled when
the model crosses ~10B parameters); use this config to override.

## How It Works

The default behavior is "smart on": every neural network you
build through `IAiModelBuilder<T,TInput,TOutput>`
runs eagerly (zero overhead) until its parameter count crosses the
threshold, at which point streaming auto-engages. To force streaming
on / off regardless of size, set `Enabled` explicitly. To
override the auto-detect threshold, set
`ThresholdParameters`.

## Properties

| Property | Summary |
|:-----|:--------|
| `Enabled` | Gets or sets whether weight streaming is enabled. |
| `ThresholdParameters` | Gets or sets the parameter-count threshold above which auto-detect engages streaming. |

