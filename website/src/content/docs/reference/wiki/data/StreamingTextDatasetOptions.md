---
title: "StreamingTextDatasetOptions"
description: "Configuration options for the streaming text dataset."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Text`

Configuration options for the streaming text dataset.

## How It Works

Streaming text datasets load text data lazily from disk, enabling training on
datasets larger than memory (e.g., The Pile, RedPajama, C4).

## Properties

| Property | Summary |
|:-----|:--------|
| `DataPath` | Root data path containing text files. |
| `FilePattern` | File pattern to match. |
| `MaxSamples` | Optional maximum number of samples to produce. |
| `Seed` | Random seed for shuffling. |
| `SequenceLength` | Sequence length for each sample. |
| `ShuffleFiles` | Shuffle files before reading. |
| `VocabularySize` | Vocabulary size for token ID bounds checking. |

