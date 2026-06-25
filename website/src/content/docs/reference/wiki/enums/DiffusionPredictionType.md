---
title: "DiffusionPredictionType"
description: "Defines what the diffusion model predicts during the denoising process."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines what the diffusion model predicts during the denoising process.

## For Beginners

When a model looks at a noisy image, it can be trained to predict:

- Epsilon (noise): "What noise was added to make this blurry?"
- Sample: "What did the clean image look like?"
- V-prediction: A mathematical blend of both (more stable training)

Most models use Epsilon prediction as it's the most common and well-studied approach.

## How It Works

Different diffusion models can be trained to predict different targets.
The prediction type affects how the scheduler interprets the model output
and computes the denoised sample.

## Fields

| Field | Summary |
|:-----|:--------|
| `Epsilon` | Model predicts the noise (epsilon) that was added to the clean sample. |
| `Sample` | Model directly predicts the clean sample (x_0). |
| `VPrediction` | Model predicts v = sqrt(alpha_cumprod) * epsilon - sqrt(1-alpha_cumprod) * x_0. |

