---
title: "ModelCompressionMode"
description: "Defines the mode of model compression to apply during serialization."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the mode of model compression to apply during serialization.

## For Beginners

Compression mode determines when and how your model gets compressed.
Like choosing between automatically archiving files vs manually selecting what to archive,
you can let the system decide the best approach or take control yourself.

## Fields

| Field | Summary |
|:-----|:--------|
| `Automatic` | The system automatically selects the best compression strategy based on model characteristics. |
| `Full` | Compresses the entire serialized model including all metadata. |
| `None` | No compression is applied. |
| `WeightsOnly` | Compresses only the model weights, leaving other metadata uncompressed. |

