---
title: "Gsm8kDataLoaderOptions"
description: "Configuration options for the GSM8K math word-problem benchmark."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Text.Benchmarks`

Configuration options for the GSM8K math word-problem benchmark.

## How It Works

GSM8K (Cobbe et al. 2021) is the canonical grade-school math benchmark
for chain-of-thought reasoning evaluation. ≈ 7,473 train / 1,319 test
problems, each with a multi-step natural-language solution and a final
numerical answer prefixed by `####`.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `MaxAnswerLength` | Maximum answer length in tokens (full chain-of-thought). |
| `MaxQuestionLength` | Maximum question length in tokens. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `Split` | Dataset split to load. |
| `VocabularySize` | Maximum vocabulary size. |

