---
title: "REBFormer<T>"
description: "REBFormer: RWKV-Enhanced E-Branchformer for efficient ASR."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.ConformerFamily`

REBFormer: RWKV-Enhanced E-Branchformer for efficient ASR.

## For Beginners

Replaces the self-attention branch in E-Branchformer with RWKV (Receptance Weighted Key Value), a linear-complexity alternative to quadratic self-attention. This enables efficient processing of long audio sequences while maintaining the parallel b...

## How It Works

**References:**

- Paper: "REB-former: RWKV-Enhanced E-Branchformer for Speech Recognition" (Song et al., 2025)

Replaces the self-attention branch in E-Branchformer with RWKV (Receptance Weighted Key Value),
a linear-complexity alternative to quadratic self-attention. This enables efficient processing
of long audio sequences while maintaining the parallel branch structure.
The RWKV branch captures global context with O(n) complexity instead of O(n^2).

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using REBFormer's RWKV-enhanced parallel branches. |

