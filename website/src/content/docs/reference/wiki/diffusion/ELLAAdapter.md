---
title: "ELLAAdapter<T>"
description: "ELLA (Efficient Large Language Model Adapter) guidance adapter for enhanced text understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Guidance`

ELLA (Efficient Large Language Model Adapter) guidance adapter for enhanced text understanding.

## For Beginners

ELLA makes the AI better at understanding complex prompts.
Instead of relying only on CLIP's text understanding, it leverages a larger
language model to capture nuances like spatial relationships and attributes.

## How It Works

ELLA bridges large language models with diffusion models by providing richer
text embeddings through an adapter network. It enhances prompt understanding
for complex, compositional descriptions without retraining the diffusion model.

Reference: Hu et al., "ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ELLAAdapter(Double)` | Initializes a new ELLA adapter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GuidanceType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(Tensor<>,Tensor<>,Double,Double)` |  |

