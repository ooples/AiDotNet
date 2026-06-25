---
title: "IModelShape"
description: "Provides shape metadata for a machine learning model, describing its expected input and output dimensions."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Provides shape metadata for a machine learning model, describing its expected input and output dimensions.

## How It Works

**For Beginners:** Every machine learning model expects data in a specific shape (size and dimensions).
For example, a model trained on images might expect input of shape [3, 224, 224] (3 color channels, 224x224 pixels),
and output a shape of [1000] (1000 class probabilities).

This interface allows models to self-describe their shapes, which is useful for:

- Validating input data before prediction
- Auto-configuring serving infrastructure
- Displaying model information without loading full weights
- Building model pipelines where output of one model feeds into another

Models can also report dynamic dimensions (e.g., variable batch size) using
`GetDynamicShapeInfo`. This follows the ONNX convention where -1
in a shape dimension means "variable at runtime".

This is an optional interface — not all models need to implement it. Base classes like
NeuralNetworkBase and ClusteringBase implement it automatically.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetDynamicShapeInfo` | Gets information about which dimensions are dynamic (variable at runtime). |
| `GetInputShape` | Gets the expected input shape of the model. |
| `GetOutputShape` | Gets the output shape of the model. |

