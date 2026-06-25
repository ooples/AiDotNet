---
title: "GatedDeltaNetLanguageModel<T>"
description: "Implements a full Gated DeltaNet language model: token embedding + N GatedDeltaNetLayer blocks + RMS norm + LM head."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements a full Gated DeltaNet language model: token embedding + N GatedDeltaNetLayer blocks + RMS norm + LM head.

## For Beginners

Gated DeltaNet improves on Mamba2 by using delta rules that can
both add and remove information from memory, leading to better sequence modeling.

## How It Works

Gated DeltaNet combines linear attention with gated delta rules for efficient sequence modeling.
The delta rule update allows the model to both write new associations and erase old ones in its
memory, unlike standard linear attention which can only accumulate.

**Reference:** Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule", 2024.

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelDimension` | Gets the model dimension (d_model). |
| `NumLayers` | Gets the number of Gated DeltaNet blocks. |
| `SupportsTraining` |  |
| `VocabSize` | Gets the vocabulary size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

