---
title: "SuperGlueDataLoaderOptions"
description: "Configuration options for the SuperGLUE benchmark data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Text.Benchmarks`

Configuration options for the SuperGLUE benchmark data loader.

## How It Works

SuperGLUE is a more challenging successor to GLUE with 8 tasks: BoolQ, CB, COPA,
MultiRC, ReCoRD, RTE, WiC, WSC.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `MaxSequenceLength` | Maximum sequence length in tokens. |
| `Split` | Dataset split to load. |
| `Task` | SuperGLUE sub-task to load. |
| `VocabularySize` | Maximum vocabulary size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

