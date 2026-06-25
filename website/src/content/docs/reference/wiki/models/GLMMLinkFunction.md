---
title: "GLMMLinkFunction"
description: "Link functions for GLMM."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Link functions for GLMM.

## Fields

| Field | Summary |
|:-----|:--------|
| `CLogLog` | Complementary log-log link: g(mu) = log(-log(1-mu)). |
| `Identity` | Identity link: g(mu) = mu. |
| `Inverse` | Inverse link: g(mu) = 1/mu. |
| `Log` | Log link: g(mu) = log(mu). |
| `Logit` | Logit link: g(mu) = log(mu/(1-mu)). |
| `Probit` | Probit link: g(mu) = Phi^-1(mu). |
| `Sqrt` | Square root link: g(mu) = sqrt(mu). |

