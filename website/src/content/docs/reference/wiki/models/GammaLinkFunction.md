---
title: "GammaLinkFunction"
description: "Link functions for Gamma regression."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Link functions for Gamma regression.

## For Beginners

The link function transforms the expected response to the scale
where linear prediction happens. Different links make different assumptions about
how predictors affect the response.

## Fields

| Field | Summary |
|:-----|:--------|
| `Identity` | Identity link: μ = Xβ. |
| `Inverse` | Inverse link: 1/μ = Xβ. |
| `Log` | Log link: ln(μ) = Xβ. |

