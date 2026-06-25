---
title: "GlueDataLoaderOptions"
description: "Configuration options for the GLUE benchmark data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Text.Benchmarks`

Configuration options for the GLUE benchmark data loader.

## How It Works

GLUE (General Language Understanding Evaluation) contains 9 NLU tasks for evaluating
language models: CoLA, SST-2, MRPC, QQP, STS-B, MNLI, QNLI, RTE, WNLI.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `MaxSequenceLength` | Maximum sequence length in tokens. |
| `Split` | Dataset split to load. |
| `Task` | GLUE sub-task to load. |
| `VocabularySize` | Maximum vocabulary size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

