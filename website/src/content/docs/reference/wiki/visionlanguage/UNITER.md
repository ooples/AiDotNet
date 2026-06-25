---
title: "UNITER<T>"
description: "UNITER (Universal Image-TExt Representation) with conditional masking pre-training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Foundational`

UNITER (Universal Image-TExt Representation) with conditional masking pre-training.

## For Beginners

UNITER is a vision-language model. Default values follow the original paper settings.

## How It Works

UNITER (Chen et al., ECCV 2020) uses a single-stream transformer with conditional masking where
either image regions or text tokens are masked during pre-training, forcing the model to learn
cross-modal alignment. Four pre-training tasks: MLM, MRM (with KL divergence), ITM, and WRA.

**References:**

- Paper: "UNITER: UNiversal Image-TExt Representation Learning" (Chen et al., ECCV 2020)

## Methods

| Method | Summary |
|:-----|:--------|
| `MeanPoolOverTokens(Tensor<>)` | Mean-pool a rank-2 [N, D] tensor over the N axis to [D]; pass through otherwise. |
| `RunStream(Tensor<>)` | Shared forward that runs the single-stream transformer and task head. |

