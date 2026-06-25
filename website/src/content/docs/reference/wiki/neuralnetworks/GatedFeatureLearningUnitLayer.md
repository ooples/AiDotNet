---
title: "GatedFeatureLearningUnitLayer<T>"
description: "Gated Feature Learning Unit (GFLU) for GANDALF architecture."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Gated Feature Learning Unit (GFLU) for GANDALF architecture.

## For Beginners

GFLU works like a smart filter:

1. Look at all features and decide which ones matter (gating)
2. Transform the selected features
3. Combine them for the next layer

The "gate" is like a dimmer switch that can turn features on/off or anywhere in between.

## How It Works

The GFLU is the core building block of GANDALF that performs feature selection
and transformation through a gating mechanism. It learns which features are
important and how to transform them.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GatedFeatureLearningUnitLayer(Int32)` | Initializes a Gated Feature Learning Unit. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Forward pass through the GFLU. |
| `GetFeatureImportance` | Gets feature importance based on gate activation magnitudes. |
| `GetGateValues` | Gets the current gate values (for interpretability). |
| `GetParameters` |  |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` |  |
| `UpdateParameters()` |  |

