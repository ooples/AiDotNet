---
title: "WikiText2DataLoaderOptions"
description: "Configuration options for the WikiText-2 data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Text.Benchmarks`

Configuration options for the WikiText-2 data loader.

## How It Works

WikiText-2 is the small-scale word-level language modeling dataset
(≈ 2M tokens train, ≈ 245k tokens validation, ≈ 281k tokens test) drawn
from verified Good and Featured Wikipedia articles. It's the canonical
fast-iteration benchmark in the LM literature — papers commonly report
WikiText-2 perplexity first (cheap), then WikiText-103 second (slow, 50× larger).

Defaults are tuned for the smaller corpus: SequenceLength=32 keeps memory
footprint modest for fast research iteration, VocabularySize=16000 covers
roughly the 95th-percentile token-frequency cutoff at this scale. Override
these for paper-grade evaluation runs.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `SequenceLength` | Sequence length for language modeling (context window). |
| `Split` | Dataset split to load. |
| `VocabularySize` | Maximum vocabulary size. |

