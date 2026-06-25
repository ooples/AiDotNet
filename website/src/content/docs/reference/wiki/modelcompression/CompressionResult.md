---
title: "CompressionResult<T>"
description: "Result of a compression operation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelCompression`

Result of a compression operation.

## Properties

| Property | Summary |
|:-----|:--------|
| `CompressedData` | The compressed data as a tensor. |
| `CompressedSizeBytes` | Compressed size in bytes (data + metadata). |
| `CompressionRatio` | Achieved compression ratio. |
| `Metadata` | Metadata required for decompression. |
| `OriginalShape` | Original shape of the data before compression. |
| `OriginalSizeBytes` | Original size in bytes. |

