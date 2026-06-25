---
title: "SerializationFormat"
description: "Specifies the serialization format used for the model payload within an AIMF envelope."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the serialization format used for the model payload within an AIMF envelope.

## How It Works

**For Beginners:** When a model is saved to disk, its internal data can be stored in different formats.
This enum tells the loader what format to expect inside the file so it can correctly read the data.

The AIMF envelope header always uses a binary format, but the actual model data inside can vary:

- Binary: Raw bytes written with BinaryWriter (most neural networks)
- Json: JSON-serialized text (clustering models, some statistical models)
- HybridBinary: A mix of binary and structured data (some complex models)

## Fields

| Field | Summary |
|:-----|:--------|
| `Binary` | Model data is stored as raw binary bytes using BinaryWriter. |
| `HybridBinary` | Model data uses a hybrid format combining binary and structured data. |
| `Json` | Model data is stored as JSON text. |

