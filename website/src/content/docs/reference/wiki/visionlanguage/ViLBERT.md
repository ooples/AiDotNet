---
title: "ViLBERT<T>"
description: "ViLBERT (Vision-and-Language BERT) with co-attention between parallel vision and language streams."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Foundational`

ViLBERT (Vision-and-Language BERT) with co-attention between parallel vision and language streams.

## For Beginners

ViLBERT is a vision-language model. Default values follow the original paper settings.

## How It Works

ViLBERT (Lu et al., NeurIPS 2019) extends BERT to a dual-stream architecture where separate
vision and language transformers interact through co-attention layers. Each co-attention layer
computes attention from one modality's queries to the other modality's keys/values.

**References:**

- Paper: "ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks" (Lu et al., NeurIPS 2019)

## Methods

| Method | Summary |
|:-----|:--------|
| `ForwardForTraining(Tensor<>)` | Same dual-stream routing as `Predict`, applied to the tape-recorded forward pass used by `TrainWithTape`. |
| `GetNamedLayerActivations(Tensor<>)` |  |
| `MeanPoolOverTokens(Tensor<>)` | Mean-pool a rank-2 [N, D] tensor over the N dimension to [D], or a rank-3 [B, N, D] tensor to [B, D]. |
| `PredictCore(Tensor<>)` | Dual-stream routing per Lu et al. |
| `RunStreamForInput(Tensor<>)` | Shared dual-stream+head routing used by both inference (`Predict`) and training (`Tensor{`): pick the stream by input shape, run it, mean-pool over the sequence/region axis, then apply the task head matching `Architecture.OutputSize`. |

