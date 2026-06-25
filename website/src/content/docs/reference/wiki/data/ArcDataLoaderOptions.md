---
title: "ArcDataLoaderOptions"
description: "Configuration options for the AI2 Reasoning Challenge (ARC) data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Text.Benchmarks`

Configuration options for the AI2 Reasoning Challenge (ARC) data loader.

## How It Works

ARC (Clark et al. 2018) is a 4-way multiple-choice grade-school science
QA benchmark. The Challenge subset contains questions that simple
retrieval/co-occurrence baselines fail; Easy subset is simpler. Both
are standard 0-shot LM-eval components.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `MaxSequenceLength` | Maximum (question + choice) sequence length in tokens. |
| `Split` | Dataset split to load. |
| `Variant` | Which ARC variant: Easy or Challenge. |
| `VocabularySize` | Maximum vocabulary size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

