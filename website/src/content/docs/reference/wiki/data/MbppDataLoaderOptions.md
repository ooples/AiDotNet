---
title: "MbppDataLoaderOptions"
description: "Configuration options for the Mostly Basic Python Problems (MBPP) loader (Austin et al."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Text.Benchmarks`

Configuration options for the Mostly Basic Python Problems (MBPP) loader (Austin et al. 2021).

## How It Works

MBPP is a 1,000-problem benchmark of basic Python programming tasks, each with
a natural-language description, canonical reference solution, and 3 unit tests.
Standard split: 974 train / 90 val / 500 test (with 100 reserved for prompt).
Used as the entry-level code-generation benchmark alongside HumanEval.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Auto-download. |
| `DataPath` | Root data path. |
| `MaxPromptLength` | Max prompt length in tokens. |
| `MaxSamples` | Optional sample cap. |
| `MaxSolutionLength` | Max solution length in tokens. |
| `Split` | Dataset split. |
| `VocabularySize` | Max vocab. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

