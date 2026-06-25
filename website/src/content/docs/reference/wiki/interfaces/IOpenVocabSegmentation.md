---
title: "IOpenVocabSegmentation<T>"
description: "Interface for open-vocabulary segmentation models that segment objects from text descriptions without being limited to a fixed set of classes."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for open-vocabulary segmentation models that segment objects from text descriptions
without being limited to a fixed set of classes.

## For Beginners

Traditional segmentation models can only recognize objects they were
trained on (e.g., "car", "person", "dog"). Open-vocabulary models can segment anything
you describe in words — even things they've never seen during training.

Example: You can ask for "red coffee mug on the table" or "person wearing a blue hat"
and the model will try to segment exactly that, even if it was never trained on those
specific combinations.

Models implementing this interface:

- SAN (CVPR 2023, side adapter on CLIP)
- CAT-Seg (CVPR 2024, cost aggregation)
- SED (CVPR 2024, simple encoder-decoder)
- Open-Vocabulary SAM (ECCV 2024, SAM + CLIP)
- Grounded SAM 2 (grounding DINO + SAM 2)
- Mask-Adapter (CVPR 2025, mask-level adaptation)

## How It Works

Open-vocabulary segmentation models use vision-language understanding (typically CLIP-based)
to segment objects described by arbitrary text. Unlike traditional models trained on a fixed
set of classes, these models can segment any concept expressible in natural language.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxCategories` | Gets the maximum number of text categories that can be queried simultaneously. |
| `MaxPromptLength` | Gets the maximum text prompt length in tokens. |

## Methods

| Method | Summary |
|:-----|:--------|
| `SegmentWithPrompt(Tensor<>,String)` | Segments an image using a single text prompt for grounded segmentation. |
| `SegmentWithText(Tensor<>,IReadOnlyList<String>)` | Segments an image using text descriptions of target classes. |

