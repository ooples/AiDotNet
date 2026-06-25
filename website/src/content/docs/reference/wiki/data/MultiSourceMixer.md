---
title: "MultiSourceMixer<TItem>"
description: "Mixes multiple data sources with configurable weights for multi-domain training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Pipeline`

Mixes multiple data sources with configurable weights for multi-domain training.

## How It Works

Combines multiple datasets into a single stream with weighted sampling.
Each batch is drawn from sources according to their mixing weights.
Commonly used for training on mixtures of domains (e.g., code + text + math).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultiSourceMixer(Int32,MultiSourceMixerOptions)` | Creates a new multi-source mixer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumSources` | Gets the number of sources. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetWeight(Int32)` | Gets the normalized weight for a source. |
| `Mix(IReadOnlyList<IEnumerator<>>,Int32)` | Mixes items from multiple sources using weighted sampling. |
| `SelectSource` | Selects which source to draw the next item from. |

