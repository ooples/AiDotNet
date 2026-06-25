---
title: "GgufReader"
description: "Parses the GGUF weight format (used by llama.cpp and the wider GGML ecosystem) into a `GgufFile`: the header, the metadata key/value store, and the tensor directory."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Agentic.Models.Local`

Parses the GGUF weight format (used by llama.cpp and the wider GGML ecosystem) into a
`GgufFile`: the header, the metadata key/value store, and the tensor directory.

## For Beginners

Hand it a `.gguf` file and it tells you the model's settings and the
weight tensors inside — the first step in loading a llama.cpp-style model.

## How It Works

Layout: magic `"GGUF"`, a version, tensor and metadata counts, the metadata KV pairs (12 value types
including typed arrays), then the tensor infos (name, dimensions, ggml type, offset), then aligned tensor
data. This reader validates the structure and exposes metadata + tensors; reading F32/F16 tensor values is
supported (quantized formats are listed but not yet dequantized). All integers are little-endian per the
spec.

## Methods

| Method | Summary |
|:-----|:--------|
| `Read(Byte[])` | Parses a GGUF file from a byte array. |
| `Read(Stream)` | Parses a GGUF file from a stream (fully read into memory). |

