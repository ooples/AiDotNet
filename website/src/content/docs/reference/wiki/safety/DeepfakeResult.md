---
title: "DeepfakeResult"
description: "Detailed result from deepfake and AI-generated image detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Image`

Detailed result from deepfake and AI-generated image detection.

## For Beginners

DeepfakeResult provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `ConsistencyScore` | Consistency analysis score (facial/spatial inconsistencies). |
| `DeepfakeScore` | Overall deepfake probability score (0.0 = authentic, 1.0 = fake). |
| `FrequencyScore` | Frequency analysis score (artifacts in frequency domain). |
| `IsDeepfake` | Whether the image is likely a deepfake or AI-generated. |
| `ProvenanceScore` | Provenance analysis score (metadata and watermark clues). |

