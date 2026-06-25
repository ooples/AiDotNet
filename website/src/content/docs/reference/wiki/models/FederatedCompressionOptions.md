---
title: "FederatedCompressionOptions"
description: "Configuration options for federated update compression (quantization, sparsification, and error feedback)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for federated update compression (quantization, sparsification, and error feedback).

## How It Works

**For Beginners:** Compression reduces the size of client updates sent to the server to save bandwidth.
This can speed up training in distributed settings, especially on slow or expensive networks.

## Properties

| Property | Summary |
|:-----|:--------|
| `Advanced` | Gets or sets advanced compression options (PowerSGD, sketching, adaptive, 1-bit SGD). |
| `QuantizationBits` | Gets or sets the number of bits used for quantization strategies. |
| `Ratio` | Gets or sets the compression ratio (0.0 to 1.0) for sparsification strategies. |
| `Strategy` | Gets or sets the compression strategy. |
| `Threshold` | Gets or sets the absolute threshold for the "Threshold" strategy. |
| `UseErrorFeedback` | Gets or sets whether to use error feedback (residual accumulation) on the client. |

