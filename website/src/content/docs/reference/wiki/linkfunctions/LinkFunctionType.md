---
title: "LinkFunctionType"
description: "Types of link functions available."
section: "API Reference"
---

`Enums` · `AiDotNet.LinkFunctions`

Types of link functions available.

## Fields

| Field | Summary |
|:-----|:--------|
| `CLogLog` | Complementary log-log link: g(μ) = log(-log(1-μ)). |
| `Identity` | Identity link: g(μ) = μ. |
| `Inverse` | Inverse link: g(μ) = 1/μ. |
| `InverseSquared` | Inverse squared link: g(μ) = 1/μ². |
| `Log` | Log link: g(μ) = log(μ). |
| `Logit` | Logit link: g(μ) = log(μ/(1-μ)). |
| `Probit` | Probit link: g(μ) = Φ⁻¹(μ). |
| `Sqrt` | Square root link: g(μ) = √μ. |

