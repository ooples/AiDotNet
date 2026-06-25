---
title: "IFullModel<T, TInput, TOutput>"
description: "Represents a complete machine learning model that combines prediction capabilities with serialization and checkpointing support."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Represents a complete machine learning model that combines prediction capabilities with serialization and checkpointing support.

## How It Works

**For Beginners:** This interface combines all the important capabilities that a complete AI model needs.

Think of IFullModel as the core contract for a machine learning model. It provides:

1. The ability to make predictions (from IModel)
- Process input data and produce output predictions

2. The ability to save and load the model (from IModelSerializer and ICheckpointableModel)
- File-based saving/loading for deployment
- Stream-based checkpointing for training resumption

3. Feature importance reporting (from IFeatureImportance)
- Understand which features contribute most to predictions

Optional capabilities (check with 'is' or InterfaceGuard before using):

- IParameterizable: Get/set model parameters (linear models, neural networks)
- IGradientComputable: Compute and apply gradients (gradient-based optimization)
- IFeatureAware: Feature selection and tracking

Not all models support all capabilities. Tree-based and ensemble models
may not implement IParameterizable or IGradientComputable.

This is particularly useful for production environments where models need to be:

- Trained once (which might take a long time)
- Saved to disk for deployment
- Checkpointed during training to prevent data loss
- Loaded quickly when needed to make predictions
- Possibly trained in a distributed manner across multiple GPUs
- Updated with new data periodically
- Used in knowledge distillation as teacher or student models

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` | Gets the default loss function used by this model for gradient computation. |

