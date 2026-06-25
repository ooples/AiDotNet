---
title: "OrdinalLinkFunction"
description: "Link functions for ordinal regression."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Link functions for ordinal regression.

## For Beginners

The link function transforms cumulative probabilities to a scale
where linear modeling can be applied. Different links make different assumptions about
how the probability changes across the ordinal scale.

## Fields

| Field | Summary |
|:-----|:--------|
| `ComplementaryLogLog` | Complementary log-log link: log(-log(1-P)). |
| `Logit` | Logit link: log(P/(1-P)). |
| `Probit` | Probit link: Φ^(-1)(P). |

