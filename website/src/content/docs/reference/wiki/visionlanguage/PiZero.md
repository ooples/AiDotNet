---
title: "PiZero<T>"
description: "pi-zero: PaliGemma VLM with action expert for 8 robot embodiments."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Robotics`

pi-zero: PaliGemma VLM with action expert for 8 robot embodiments.

## For Beginners

pi-zero is a vision-language-action model for general robot
control across multiple robot platforms. Default values follow the original paper settings.

## How It Works

pi-zero (Black et al., 2024) is a vision-language-action flow model built on PaliGemma VLM
with a dedicated action expert for general robot control. It uses flow matching to generate
continuous robot actions, trained across 8 different robot embodiments for dexterous
manipulation including folding, assembly, and bin-picking tasks.

**References:**

- Paper: "pi0: A Vision-Language-Action Flow Model for General Robot Control (Black et al., 2024)"

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates from image using pi-zero's PaliGemma VLM backbone + flow matching conditioning. |
| `PredictAction(Tensor<>,String)` | Predicts action using pi-zero's flow matching formulation. |

