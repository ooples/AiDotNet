---
title: "SafetensorsFile"
description: "A parsed safetensors file: the tensor table of contents plus access to each tensor's raw bytes and a conversion to `Double` values."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models.Local`

A parsed safetensors file: the tensor table of contents plus access to each tensor's raw bytes and a
conversion to `Double` values. This is the format-level loading primitive; mapping the loaded
tensors onto a specific network's layers is architecture-specific.

## For Beginners

The opened file. Ask it which tensors it has, get their shapes, and read any
one as an array of numbers — the raw material for loading pretrained weights into a model.

## How It Works

Supports the common float dtypes `F64`, `F32`, and `F16` for value conversion; other dtypes
are still listed and their raw bytes are accessible. Values are read little-endian (the safetensors
convention).

The file may be backed by an in-memory byte array or by a seekable stream
(see `Stream)`). When stream-backed, tensor bytes are read on demand —
the stream must remain open for the lifetime of this instance, and concurrent reads are serialized
internally.

## Properties

| Property | Summary |
|:-----|:--------|
| `Names` | Gets the names of all tensors. |
| `TensorNames` |  |
| `Tensors` | Gets the tensors in the file. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Get(String)` | Returns the descriptor for a tensor, or `null` when not present. |
| `GetRawBytes(String)` | Copies a tensor's raw bytes. |
| `ReadAsDouble(String)` | Reads a tensor's values as `Double` (supports F64, F32, F16). |
| `ReadTensorBytes(SafetensorsTensor,Nullable<Int32>)` | Centralized span validation + read: every raw/decode path goes through here, so no code touches the backing data with unvalidated metadata-derived offsets. |

