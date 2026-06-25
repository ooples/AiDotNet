---
title: "PennTreebankDataLoaderOptions"
description: "Configuration options for the Penn Treebank (PTB) data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Text.Benchmarks`

Configuration options for the Penn Treebank (PTB) data loader.

## How It Works

PTB is the classic small-scale word-level LM benchmark — ≈ 887k tokens train,
≈ 70k val, ≈ 79k test, vocab ≈ 10k after preprocessing. Most pre-2018 LM
papers report PTB perplexity. Useful as a sanity check before scaling.
Tokenization follows the Mikolov-style preprocessed split.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `SequenceLength` | Sequence length (BPTT context). |
| `Split` | Dataset split to load. |
| `VocabularySize` | Maximum vocabulary size. |

