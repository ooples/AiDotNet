---
title: "RepresentationalBiasDetector<T>"
description: "Detects representational bias by analyzing whether demographic groups are underrepresented, overrepresented, or systematically associated with specific roles/contexts in text."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Fairness`

Detects representational bias by analyzing whether demographic groups are underrepresented,
overrepresented, or systematically associated with specific roles/contexts in text.

## For Beginners

Imagine a children's book where all the scientists are men and
all the teachers are women. Even if nothing negative is said, the lopsided representation
teaches children that certain roles "belong" to certain groups. This detector finds
exactly that kind of imbalance in AI-generated text.

## How It Works

Representational bias occurs when certain groups are disproportionately represented in
specific contexts. For example, if "doctor" almost always appears near male pronouns
and "nurse" near female pronouns, the text reinforces occupational representation bias.
This module counts demographic group mentions and measures role-association imbalances.

**References:**

- BEATS: Comprehensive bias evaluation test suite for LLMs (2025, arxiv:2503.24310)
- Representation bias in text generation (Sheng et al., EMNLP 2019)
- Measuring representational harms in language technology (Blodgett et al., 2020)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RepresentationalBiasDetector(Double,String[])` | Initializes a new representational bias detector. |

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

