---
title: "AgNewsDataLoaderOptions"
description: "Configuration options for the AG News topic classification data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Text.Benchmarks`

Configuration options for the AG News topic classification data loader.

## How It Works

AG News is a 4-class news topic classification dataset (World, Sports,
Business, Sci/Tech) — 120k training examples, 7.6k test examples. The
canonical small classification benchmark from Zhang et al. 2015.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `MaxSequenceLength` | Maximum sequence length per example. |
| `Split` | Dataset split to load. |
| `VocabularySize` | Maximum vocabulary size. |

