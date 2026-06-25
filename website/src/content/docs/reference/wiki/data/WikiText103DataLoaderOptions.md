---
title: "WikiText103DataLoaderOptions"
description: "Configuration options for the WikiText-103 data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Text.Benchmarks`

Configuration options for the WikiText-103 data loader.

## How It Works

WikiText-103 is a large word-level language modeling dataset with over 100M tokens
from verified Good and Featured Wikipedia articles. Used for language model training/evaluation.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `SequenceLength` | Sequence length for language modeling (context window). |
| `Split` | Dataset split to load. |
| `VocabularySize` | Maximum vocabulary size. |

