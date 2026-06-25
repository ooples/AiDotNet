---
title: "SafeTensorsLoader<T>"
description: "Loads model weights from SafeTensors format files."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelLoading`

Loads model weights from SafeTensors format files.

## For Beginners

SafeTensors is like a special container for AI model weights.

Why SafeTensors instead of pickle files?

- Safe: Cannot execute arbitrary code (unlike pickle)
- Fast: Memory-mapped loading for quick access
- Simple: Just tensors and their metadata

This loader reads SafeTensors files and converts them to our Tensor format
so we can use pretrained weights from HuggingFace and other sources.

File structure:
```
[8 bytes: header length]
[JSON header: tensor metadata]
[tensor data: raw bytes]
```

## How It Works

SafeTensors is a format developed by Hugging Face for storing model tensors safely.
It's the standard format for Stable Diffusion and other modern ML models.

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateTensor(Byte[],String,Int32[])` | Creates a Tensor from raw bytes and metadata. |
| `GetTensorInfo(String)` | Gets the list of tensor names in a SafeTensors file without loading data. |
| `Load(String)` | Loads all tensors from a SafeTensors file. |
| `Load(String,IEnumerable<String>)` | Loads specific tensors from a SafeTensors file. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides numeric operations for the specific type T. |

