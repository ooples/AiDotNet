---
title: "PretrainedModelLoader<T>"
description: "Loads pretrained models from various sources."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelLoading`

Loads pretrained models from various sources.

## For Beginners

This is your gateway to using pretrained models.

Instead of training models from scratch (which requires massive datasets and
compute resources), you can load pretrained weights that others have trained.

Example usage:
```cs
var loader = new PretrainedModelLoader<float>();

// Load a pretrained VAE
var vae = new StandardVAE<float>();
await loader.LoadVAEWeights(vae, "sd-vae-ft-mse/diffusion_pytorch_model.safetensors");

// Now your VAE is ready for image encoding/decoding!
```

## How It Works

This class provides methods to load pretrained model weights into our model classes.
It supports loading from local SafeTensors files and handles weight name mapping
between different model formats.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PretrainedModelLoader(Boolean)` | Initializes a new instance of the PretrainedModelLoader class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetTensorInfo(String)` | Gets information about tensors in a SafeTensors file. |
| `LoadAllTensors(String)` | Loads all tensors from a SafeTensors file. |
| `LoadTensors(String,IEnumerable<String>)` | Loads specific tensors by name from a SafeTensors file. |
| `LoadWeights(IWeightLoadable<>,String,Func<String,String>,Boolean)` | Loads weights from a SafeTensors file into any IWeightLoadable model. |
| `LoadWeights(IWeightLoadable<>,String,WeightMapping,Boolean)` | Loads weights using a WeightMapping instance. |
| `ValidateWeights(String,IEnumerable<String>)` | Validates that required tensors exist in a weights file. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides numeric operations for the specific type T. |
| `_safeTensorsLoader` | The SafeTensors loader instance. |
| `_verbose` | Whether to log loading progress. |

