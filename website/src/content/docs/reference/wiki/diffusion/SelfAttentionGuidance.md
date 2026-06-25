---
title: "SelfAttentionGuidance<T>"
description: "Self-Attention Guidance (SAG) for diffusion model inference."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Guidance`

Self-Attention Guidance (SAG) for diffusion model inference.

## For Beginners

SAG looks at which parts of the image the model pays
most attention to, then uses that information to guide generation more
precisely. It can improve detail in important areas while reducing artifacts.

## How It Works

SAG leverages self-attention maps to selectively blur high-attention regions,
creating an intermediate prediction between conditional and unconditional.
This provides more focused guidance than standard CFG.

Reference: Hong et al., "Improving Sample Quality of Diffusion Models Using Self-Attention Guidance", ICCV 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SelfAttentionGuidance(Double,Double)` | Initializes a new Self-Attention Guidance instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GuidanceType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(Tensor<>,Tensor<>,Double,Double)` |  |

