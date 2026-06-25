---
title: "DomainMixingSampler"
description: "Samples from multiple data domains with configurable mixing ratios for multi-domain LLM training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Sampling`

Samples from multiple data domains with configurable mixing ratios for multi-domain LLM training.

## How It Works

Domain mixing controls what proportion of each batch comes from different data sources
(e.g., 40% web, 30% code, 20% books, 10% academic). This is the technique used
by Llama, GPT-4, and other large language models to balance data quality and diversity.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DomainMixingSampler(Int32[],Double[],Nullable<Int32>)` | Initializes a new domain mixing sampler. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Length` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetIndicesCore` | Returns indices sampled according to domain mixing weights. |

