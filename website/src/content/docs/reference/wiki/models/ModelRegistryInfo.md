---
title: "ModelRegistryInfo<T, TInput, TOutput>"
description: "Contains structured model registry information from a trained model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Contains structured model registry information from a trained model.

## For Beginners

This is a container for model versioning and registry data.

It includes:

- The registered model name and version number
- Access to the registry for stage transitions (Staging -> Production)
- Checkpoint path for loading/saving the model
- Access to the checkpoint manager for persistence operations

## How It Works

This record provides type-safe access to model registry data, including
the registered model name, version, and access to the registry for lifecycle management.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ModelRegistryInfo(String,Nullable<Int32>,IModelRegistry<,,>,String,ICheckpointManager<,,>)` | Contains structured model registry information from a trained model. |

