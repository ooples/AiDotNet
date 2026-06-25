---
title: "BetaLinkFunction"
description: "Link functions for Beta Regression mean model."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Link functions for Beta Regression mean model.

## Fields

| Field | Summary |
|:-----|:--------|
| `CLogLog` | Complementary log-log link: η = log(-log(1 - μ)). |
| `Log` | Log link: η = log(μ). |
| `Logit` | Logit link: η = log(μ / (1 - μ)). |
| `Probit` | Probit link: η = Φ⁻¹(μ). |

