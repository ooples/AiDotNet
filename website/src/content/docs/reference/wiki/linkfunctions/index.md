---
title: "Link Functions"
description: "All 11 public types in the AiDotNet.linkfunctions namespace, organized by kind."
section: "API Reference"
---

**11** public types in this namespace, organized by kind.

## Models & Types (8)

| Type | Summary |
|:-----|:--------|
| [`CLogLogLink<T>`](/docs/reference/wiki/linkfunctions/clogloglink/) | Complementary log-log link function: g(μ) = log(-log(1-μ)). |
| [`IdentityLink<T>`](/docs/reference/wiki/linkfunctions/identitylink/) | Identity link function: g(μ) = μ. |
| [`InverseSquaredLink<T>`](/docs/reference/wiki/linkfunctions/inversesquaredlink/) | Inverse squared link function: g(mu) = 1/mu^2. |
| [`LogLink<T>`](/docs/reference/wiki/linkfunctions/loglink/) | Log link function: g(μ) = log(μ). |
| [`LogitLink<T>`](/docs/reference/wiki/linkfunctions/logitlink/) | Logit link function: g(μ) = log(μ/(1-μ)). |
| [`ProbitLink<T>`](/docs/reference/wiki/linkfunctions/probitlink/) | Probit link function: g(μ) = Φ⁻¹(μ), where Φ is the standard normal CDF. |
| [`ReciprocalLink<T>`](/docs/reference/wiki/linkfunctions/reciprocallink/) | Inverse (reciprocal) link function: g(μ) = 1/μ. |
| [`SqrtLink<T>`](/docs/reference/wiki/linkfunctions/sqrtlink/) | Square root link function: g(μ) = √μ. |

## Enums (2)

| Type | Summary |
|:-----|:--------|
| [`GlmDistributionFamily`](/docs/reference/wiki/linkfunctions/glmdistributionfamily/) | Distribution families for GLMs. |
| [`LinkFunctionType`](/docs/reference/wiki/linkfunctions/linkfunctiontype/) | Types of link functions available. |

## Helpers & Utilities (1)

| Type | Summary |
|:-----|:--------|
| [`LinkFunctionFactory<T>`](/docs/reference/wiki/linkfunctions/linkfunctionfactory/) | Factory for creating link function instances. |

