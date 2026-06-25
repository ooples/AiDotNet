---
title: "SafetensorsReader"
description: "Parses the `` weight format — the safe, simple format used to distribute pretrained model weights — into a `SafetensorsFile`."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Agentic.Models.Local`

Parses the `` weight format — the
safe, simple format used to distribute pretrained model weights — into a `SafetensorsFile`.
This is the loading primitive for bringing external weights into AiDotNet's local engine.

## For Beginners

Hand it a `.safetensors` file (bytes or a stream) and it tells you every
weight array inside and lets you read them — the first step in loading a downloaded model.

## How It Works

Layout: an 8-byte little-endian header length, a JSON header mapping tensor names to
`{dtype, shape, data_offsets}` (plus an optional `__metadata__`), then the concatenated tensor
bytes. This reader validates the framing and exposes each tensor; mapping the tensors to a particular
network's parameters (which depends on the architecture's layer naming) is a separate step.

## Methods

| Method | Summary |
|:-----|:--------|
| `Read(Byte[])` | Parses a safetensors file from a byte array. |
| `Read(Stream)` | Parses a safetensors file from a stream. |

