---
title: "TweedieLinkFunction"
description: "Link functions for Tweedie regression."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Link functions for Tweedie regression.

## For Beginners

The link function transforms the expected response to the scale
where linear prediction happens. Different links make different assumptions about
how predictors affect the response.

## Fields

| Field | Summary |
|:-----|:--------|
| `Identity` | Identity link: μ = Xβ. |
| `Log` | Log link: ln(μ) = Xβ. |
| `Power` | Power link: μ^(1-p) = Xβ. |

