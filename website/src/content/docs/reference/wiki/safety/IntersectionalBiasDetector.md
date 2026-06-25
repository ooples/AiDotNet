---
title: "IntersectionalBiasDetector<T>"
description: "Detects intersectional bias — bias that uniquely affects individuals at the intersection of multiple demographic identities (e.g., Black women, elderly Asian men)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Fairness`

Detects intersectional bias — bias that uniquely affects individuals at the intersection
of multiple demographic identities (e.g., Black women, elderly Asian men).

## For Beginners

A person is not just their gender or their race — they are both at
the same time. A Black woman may face unique biases that differ from what Black men or
White women experience. This detector looks for cases where AI output is specifically
biased against people with multiple overlapping identities.

## How It Works

Intersectional bias occurs when the combined effect of belonging to multiple demographic
groups produces worse outcomes than would be predicted by looking at each group individually.
For example, a model might generate positive text about "women" and about "Black people"
separately, but generate negative text about "Black women" specifically. This module detects
such compounding bias by analyzing sentiment around intersectional identity mentions.

**References:**

- Demarginalizing the intersection of race and sex (Crenshaw, 1989)
- Intersectional bias in AI fairness (Buolamwini & Gebru, Gender Shades, 2018)
- BEATS: Comprehensive bias evaluation test suite for LLMs (2025, arxiv:2503.24310)
- FLEX: Robustness of fairness evaluation under adversarial prompts (NAACL 2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IntersectionalBiasDetector(Double)` | Initializes a new intersectional bias detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsReady` |  |
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Vector<>)` |  |
| `EvaluateText(String)` |  |

