---
title: "AdversarialImageEvaluator<T>"
description: "Detects adversarial perturbations in images via the learnable feature-squeezing ensemble described by Xu et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Adversarial`

Detects adversarial perturbations in images via the learnable feature-squeezing
ensemble described by Xu et al. 2018.

## For Beginners

An adversarial image looks normal to humans but tricks AI
classifiers. This module looks for three telltale patterns of injected noise and
learns the right weight on each (some attacks reveal themselves more in HF energy,
others in pixel-histogram gaps).

## How It Works

Adversarial attacks add imperceptible perturbations that cause classifiers to
misclassify content. This module detects such perturbations by extracting three
statistical features from the image and combining them with a learnable linear
ensemble:

Per Xu et al. 2018 §4 the final score is a learnable weighted combination of these
detectors (the paper describes it as a logistic ensemble). This implementation models
that as a single `DenseLayer` mapping `[B, 3]` features → `[B, 1]`
score, with a sigmoid activation for the [0, 1] range that the
`Tensor{` threshold expects.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdversarialImageEvaluator(Double)` | Initializes a new adversarial image evaluator. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsReady` |  |
| `ModuleName` |  |
| `ParameterCount` | AIE's pipeline always emits a 3-element feature vector to a single `DenseLayer(3 → 1)`: `3 × 1 = 3` weights + `1` bias = 4 learnable parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Evaluate(Vector<>)` |  |
| `EvaluateImage(Tensor<>)` |  |
| `GetModelMetadata` |  |
| `GetNamedLayerActivations(Tensor<>)` | AIE's `Predict` doesn't feed the input image directly into `Layers[0]` — it first extracts a 3-element feature vector (`Int32[])`, `ReadOnlySpan{`, `ReadOnlySpan{`) and only then runs the Dense(3 → 1) head. |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `UpdateParameters(Vector<>)` |  |

