---
title: "CLIPA<T>"
description: "CLIPA (CLIP with Inverse scaling law and Accelerated training) model using progressive resolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

CLIPA (CLIP with Inverse scaling law and Accelerated training) model using progressive resolution.

## For Beginners

CLIPA discovers that training CLIP at low resolution first,
then fine-tuning at full resolution, reduces training cost by 7-8x while maintaining
performance. This "inverse scaling law" makes it much cheaper to train strong vision-
language models by progressively increasing image resolution during training. Default
values follow the original paper settings.

## How It Works

CLIPA (Li et al., 2023) discovers an inverse scaling law: training at lower resolution first then
fine-tuning at full resolution reduces training cost 7-8x while maintaining performance.

**References:**

- Paper: "An Inverse Scaling Law for CLIP Training" (Li et al., 2023)

**Architecture layout:** Vision encoder (patch embedding + transformer + projection)
lives in `Layers`; text encoder lives in
`TextEncoderLayers`. Default
`Predict` + `Tensor{` is vision-only;
`String)` walks the text stack independently.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetExtraTrainableLayers` |  |
| `SyncImageSizeWithArchitecture` | Aligns `_options.ImageSize` with `Architecture.InputHeight` when the architecture declares an explicit square spatial extent. |

