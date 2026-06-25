---
title: "TransformerState<T>"
description: "Represents the serializable state of a fitted time series transformer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Represents the serializable state of a fitted time series transformer.

## For Beginners

This class allows you to save a fitted transformer to a file
and reload it later without needing to re-fit. This is useful for:

- Production deployments where you train once and deploy the fitted model
- Sharing trained transformers between team members
- Versioning and reproducibility of your feature engineering pipeline

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureNames` | The generated feature names. |
| `IncrementalState` | The incremental state, if initialized. |
| `InputFeatureCount` | The number of input features. |
| `InputFeatureNames` | The input feature names. |
| `IsFitted` | Whether the transformer has been fitted. |
| `Options` | The serialized options used to configure the transformer. |
| `OutputFeatureCount` | The number of output features. |
| `Parameters` | Transformer-specific parameters and learned values. |
| `TransformerType` | The type name of the transformer. |
| `Version` | The version of the serialization format. |
| `WindowSizes` | The window sizes used by the transformer. |

