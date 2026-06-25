---
title: "ErrorFeedbackCompressor<T>"
description: "Error feedback compressor: wraps any compression method with residual accumulation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Compression`

Error feedback compressor: wraps any compression method with residual accumulation.

## For Beginners

Most compression methods are biased — they lose some information.
Error feedback fixes this by remembering what was lost (the "residual") and adding it back
in the next round. Over time, all information eventually gets transmitted, making the
compressed training converge to the same solution as uncompressed training.

## How It Works

**How it works:**

**Mathematical guarantee:** With error feedback, any contractive compressor
(compression preserves a fraction of the signal) converges at the same rate as
uncompressed SGD, up to a constant factor.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ErrorFeedbackCompressor(AdvancedCompressionOptions)` | Initializes a new instance of `ErrorFeedbackCompressor`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyErrorFeedback(Tensor<>,Int32)` | Applies error feedback to a gradient before compression. |
| `CompressOneBit(Tensor<>,Int32)` | Applies 1-bit SGD compression (sign-only encoding) with error feedback. |
| `GetErrorNorm(Int32)` | Gets the accumulated error norm for a client (for monitoring convergence). |
| `ResetAllErrors` | Resets all accumulated errors across all clients. |
| `ResetError(Int32)` | Resets the accumulated error for a specific client. |
| `UpdateError(Tensor<>,Tensor<>,Int32)` | Updates the error accumulator after compression. |

