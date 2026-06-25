---
title: "SelfConditionedCTC<T>"
description: "Self-Conditioned CTC: iterative refinement through encoder layers"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.CTCVariants`

Self-Conditioned CTC: iterative refinement through encoder layers

## For Beginners

Self-Conditioned CTC feeds CTC alignment predictions from intermediate layers back into subsequent encoder layers as conditioning information. Each encoder block receives both the audio features and the CTC posterior distribution from the previous...

## How It Works

**References:**

- Paper: "Self-Conditioned CTC: CTC Alignment Conditioning on Intermediate Predictions" (Nozaki and Komatsu, 2021)

Self-Conditioned CTC feeds CTC alignment predictions from intermediate layers back into subsequent encoder layers as conditioning information. Each encoder block receives both the audio features and the CTC posterior distribution from the previous block's auxiliary CTC head. This self-conditioning enables iterative refinement: lower layers produce rough alignments that upper layers refine. The technique significantly reduces CTC's conditional independence assumption limitation.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using self-conditioned CTC iterative refinement. |

