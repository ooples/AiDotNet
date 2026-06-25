---
title: "InverseGaussianLinkFunction"
description: "Link functions for Inverse Gaussian regression."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Link functions for Inverse Gaussian regression.

## For Beginners

The link function transforms the expected response to the scale
where linear prediction happens. Different links make different assumptions about
how predictors affect the response.

## Fields

| Field | Summary |
|:-----|:--------|
| `Identity` | Identity link: μ = Xβ. |
| `Inverse` | Inverse link: 1/μ = Xβ. |
| `InverseSquared` | Inverse squared link: -1/(2μ²) = Xβ. |
| `Log` | Log link: ln(μ) = Xβ. |

