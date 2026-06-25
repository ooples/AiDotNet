---
title: "OpenVocabSegmentationOutput<T>"
description: "Output for open-vocabulary and referring segmentation models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Common`

Output for open-vocabulary and referring segmentation models.

## For Beginners

Open-vocabulary segmentation lets you segment objects described by
arbitrary text. This output maps each text query to its corresponding mask(s) in the image.
Referring segmentation adds reasoning capability — the model can also explain what it found.

## Properties

| Property | Summary |
|:-----|:--------|
| `BoundingBoxes` | Per-query bounding boxes [numQueries, 4] as (x1, y1, x2, y2). |
| `GroundingTokens` | Grounding tokens connecting text spans to image regions (for grounded models like GLaMM). |
| `InferenceTime` | Inference time. |
| `Masks` | Per-query masks [numQueries, H, W]. |
| `Queries` | Text queries that were used. |
| `Scores` | Per-query confidence scores. |
| `SemanticMap` | Combined semantic map [H, W] where each pixel is assigned to the best-matching query. |
| `SimilarityScores` | Text-image similarity scores [numQueries] indicating how well each query matches the image. |
| `TextResponse` | Model's textual response (for referring/reasoning segmentation models like LISA). |

