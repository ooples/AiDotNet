---
title: "TabNetNetwork<T>"
description: "TabNet neural network for interpretable tabular learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

TabNet neural network for interpretable tabular learning.

## For Beginners

TabNet is designed for interpretability:

Architecture:

1. **Feature Transformer**: Shared layers process all features
2. **Attentive Transformer**: Selects which features to use at each step
3. **Decision Steps**: Multiple rounds of feature selection
4. **Sparse Attention**: Only a few features are used per step

Key insight: At each decision step, TabNet decides "which features should I
focus on?" This sequential attention makes the model interpretable - you can
see exactly which features were used for each prediction.

TabNet often matches gradient boosting (XGBoost, LightGBM) while providing
built-in feature importance and interpretability.

## How It Works

TabNet uses sequential attention to choose which features to reason from at each decision step,
enabling interpretable feature selection while achieving performance competitive with gradient boosting.
This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Engine-based tensor operations for CPU/GPU acceleration
- Full autodiff and JIT compilation support

Reference: "TabNet: Attentive Interpretable Tabular Learning" (Arik & Pfister, AAAI 2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabNetNetwork` | Initializes a new instance with default architecture settings. |
| `TabNetNetwork(NeuralNetworkArchitecture<>,TabNetOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Double)` | Initializes a new TabNet network with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureDimension` | Gets the feature dimension. |
| `NumDecisionSteps` | Gets the number of decision steps. |
| `Options` | Gets the TabNet-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

