---
title: "DeepLIFTExplainer<T>"
description: "DeepLIFT (Deep Learning Important FeaTures) explainer for neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

DeepLIFT (Deep Learning Important FeaTures) explainer for neural networks.

## For Beginners

DeepLIFT is a method for explaining neural network predictions
by comparing activations to a reference/baseline.

How it differs from gradients:

- Gradients: "How would the output change if I slightly changed the input?"
- DeepLIFT: "How much does each input contribute compared to a baseline?"

Key concepts:

1. **Reference/Baseline**: A neutral input (like zeros or average input)
2. **Difference from reference**: Compares actual activations to reference activations
3. **Multipliers**: How much each neuron's difference-from-reference contributes

DeepLIFT variants:

- **Rescale**: Distributes contribution proportionally
- **RevealCancel**: Handles positive and negative contributions separately

Advantages over gradients:

- More stable than gradients (no saturation issues)
- Handles non-linearities better (ReLU, etc.)
- Contributions sum to the difference between output and baseline output

Example: For a spam classifier:

- Reference: Average email or neutral text
- DeepLIFT shows which words made the email MORE or LESS likely to be spam

compared to the reference

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepLIFTExplainer(Func<Vector<>,Vector<>>,Func<Vector<>,Vector<>>,Func<Vector<>,Vector<>,Vector<>>,Int32,Vector<>,String[],DeepLIFTRule)` | Initializes a new DeepLIFT explainer. |
| `DeepLIFTExplainer(INeuralNetwork<>,Int32,Vector<>,String[],DeepLIFTRule)` | Initializes a new DeepLIFT explainer from a neural network model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsGPUAccelerated` |  |
| `MethodName` |  |
| `SupportsGlobalExplanations` |  |
| `SupportsLocalExplanations` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeApproximateAttributions(Vector<>,Vector<>,Int32)` | Computes approximate DeepLIFT attributions using path integration. |
| `ComputeAttributionsFromMultipliers(Vector<>,Vector<>,Vector<>)` | Computes attributions from multipliers. |
| `ComputeNumericalGradient(Vector<>,Int32)` | Computes gradient using backpropagation (if available) or numerical approximation. |
| `Explain(Vector<>)` | Computes DeepLIFT attributions for an input. |
| `Explain(Vector<>,Int32)` | Computes DeepLIFT attributions for a specific output. |
| `ExplainBatch(Matrix<>)` |  |
| `SetGPUHelper(GPUExplainerHelper<>)` |  |

