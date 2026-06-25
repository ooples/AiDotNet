---
title: "GgufFile"
description: "A parsed GGUF file: its version, the metadata key/value store (hyperparameters, tokenizer config, etc.), the tensor directory, and access to F32/F16 tensor values."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models.Local`

A parsed GGUF file: its version, the metadata key/value store (hyperparameters, tokenizer config, etc.),
the tensor directory, and access to F32/F16 tensor values. This is the structural loading primitive for
GGUF (the llama.cpp weight format).

## For Beginners

The opened GGUF file. Ask it for the model's settings (metadata), which weight
tensors it has, and read the float ones as numbers.

## How It Works

Metadata values are exposed as .NET objects (numbers, strings, bools, or arrays). F32 and F16 tensors can
be read as `Double`; quantized tensors (Q4_K, Q8_0, …) are listed with their type/offset but
dequantization is a follow-up — their raw bytes are still available.

## Properties

| Property | Summary |
|:-----|:--------|
| `Alignment` | Gets the tensor-data alignment. |
| `Metadata` | Gets the metadata key/value store. |
| `TensorNames` |  |
| `Tensors` | Gets the tensor directory. |
| `Version` | Gets the GGUF format version. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Get(String)` | Returns the tensor info for a name, or `null` when not present. |
| `GetMetadata(String)` | Gets a metadata value, or `null` when the key is absent. |
| `ReadAsDouble(String)` | Reads an F32 or F16 tensor's values as `Double`. |
| `ValidateBlockSpan(GgufTensorInfo,Int32,Int32,Int32)` | Validates a quantized tensor's block alignment and byte span, returning its absolute offset. |
| `ValidateSpan(GgufTensorInfo,Int32,Int64,Int32)` | Centralized span validation: every decode path goes through here (directly or via `Int32)`) before touching `_data`, so a malformed GGUF directory can never drive a read past the payload or silently decode a partially-filled tensor. |

