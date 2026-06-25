---
title: "EBranchformer<T>"
description: "E-Branchformer: enhanced Branchformer with improved merging"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.CTCVariants`

E-Branchformer: enhanced Branchformer with improved merging

## For Beginners

E-Branchformer enhances the Branchformer architecture with an improved merging strategy. Instead of simple concatenation and gating, E-Branchformer uses a point-wise feed-forward module with depthwise convolution for branch merging. The enhanced m...

## How It Works

**References:**

- Paper: "E-Branchformer: Branchformer with Enhanced merging for speech recognition" (Kim et al., 2023)

E-Branchformer enhances the Branchformer architecture with an improved merging strategy. Instead of simple concatenation and gating, E-Branchformer uses a point-wise feed-forward module with depthwise convolution for branch merging. The enhanced merge module allows richer interaction between the attention and convolution branch outputs. E-Branchformer achieves the best reported results among non-Whisper models on LibriSpeech and is the default encoder in ESPnet.

## Methods

| Method | Summary |
|:-----|:--------|
| `TokensToText(List<Int32>)` | Maps token IDs to text. |
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using E-Branchformer's enhanced parallel-branch encoder. |

