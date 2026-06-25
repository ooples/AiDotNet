---
title: "HellaswagDataLoaderOptions"
description: "Configuration options for the HellaSwag commonsense NLI benchmark (Zellers et al."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Text.Benchmarks`

Configuration options for the HellaSwag commonsense NLI benchmark
(Zellers et al. 2019).

## How It Works

HellaSwag is a 4-way multiple-choice commonsense reasoning benchmark
where each example presents a context sentence and 4 possible endings;
only one is the natural continuation. ≈ 39,905 train / 10,042 val
problems. Adversarially filtered against early LMs — remains one of
the standard 0-shot LM-eval benchmarks today.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `MaxSequenceLength` | Maximum sequence length per (context + ending) pair. |
| `Split` | Dataset split to load. |
| `VocabularySize` | Maximum vocabulary size. |

