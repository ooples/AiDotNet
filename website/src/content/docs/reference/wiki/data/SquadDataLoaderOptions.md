---
title: "SquadDataLoaderOptions"
description: "Configuration options for the SQuAD data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Text.Benchmarks`

Configuration options for the SQuAD data loader.

## How It Works

SQuAD (Stanford Question Answering Dataset) contains 100K+ question-answer pairs
on Wikipedia articles. SQuAD 2.0 adds 50K unanswerable questions.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `MaxContextLength` | Maximum context length in tokens. |
| `MaxQuestionLength` | Maximum question length in tokens. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `Split` | Dataset split to load. |
| `Version2` | Use SQuAD 2.0 (includes unanswerable questions). |
| `VocabularySize` | Maximum vocabulary size. |

