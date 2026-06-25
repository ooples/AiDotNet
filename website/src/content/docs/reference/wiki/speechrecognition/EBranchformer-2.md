---
title: "EBranchformer<T>"
description: "E-Branchformer (Enhanced Branchformer) speech recognition model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.ConformerFamily`

E-Branchformer (Enhanced Branchformer) speech recognition model.

## For Beginners

Extends Branchformer with an enhanced merge module that applies depthwise convolution before the concatenation-merge, improving local-global fusion. The merge module output is: concat(attn_branch, cgmlp_branch) -> depthwise conv -> linear projecti...

## How It Works

**References:**

- Paper: "E-Branchformer: Branchformer with Enhanced Merging" (Kim et al., 2022)

Extends Branchformer with an enhanced merge module that applies depthwise convolution
before the concatenation-merge, improving local-global fusion. The merge module output
is: concat(attn_branch, cgmlp_branch) -> depthwise conv -> linear projection -> residual.
Achieves SOTA on LibriSpeech with ESPnet (WER 2.1%/4.2% clean/other).

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using E-Branchformer's enhanced parallel-branch encoder. |

