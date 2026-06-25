---
title: "DFNCLIP<T>"
description: "DFN-CLIP (Data Filtering Networks for CLIP) model using filtered high-quality training data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

DFN-CLIP (Data Filtering Networks for CLIP) model using filtered high-quality training data.

## For Beginners

DFN-CLIP improves CLIP by using a small trained CLIP model to
score and filter image-text pairs from a massive noisy dataset, keeping only high-quality
pairs. The larger model trained on this filtered data achieves 83.0% zero-shot ImageNet
accuracy with ViT-H/14, demonstrating that data quality matters more than quantity.
Default values follow the original paper settings.

## How It Works

DFN-CLIP (Fang et al., 2023) uses a small CLIP model to score and filter image-text pairs
from a large noisy pool, then trains a larger model on only high-quality data. Achieves 83.0%
zero-shot on ImageNet with ViT-H/14.

**References:**

- Paper: "Data Filtering Networks" (Fang et al., 2023)

**Architecture layout:** Mirrors the PyTorch / HuggingFace CLIP layout —
vision encoder (patch embedding + transformer + projection) lives in
`Layers` so the default forward and tape-based
training paths walk it correctly. The text encoder lives in a separate field
surfaced through `GetExtraTrainableLayers` so it participates in
streaming offload / weight-registry hooks but isn't on the vision-only forward
path. Real contrastive training is done through paired-data APIs
(`String)` / `String[])`); the
inherited `Predict` + `Tensor{`
surface is the vision-encoder-only path that lets CLIP plug into
generic NN consumers without crashing on cross-stack layer walks.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetExtraTrainableLayers` | Surface the text-encoder stack to streaming-offload / weight-registry hooks (NeuralNetworkBase.GetExtraTrainableLayers contract). |
| `PredictCore(Tensor<>)` | Vision-only forward: walks `Layers` (patch embedding → vision transformer → vision projection) on the preprocessed image. |
| `SyncImageSizeWithArchitecture` | Aligns `_options.ImageSize` with `Architecture.InputHeight` when the architecture declares an explicit square spatial extent. |

