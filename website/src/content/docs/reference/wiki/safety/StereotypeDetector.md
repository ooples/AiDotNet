---
title: "StereotypeDetector<T>"
description: "Detects stereotypical associations between demographic groups and attributes in text."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Fairness`

Detects stereotypical associations between demographic groups and attributes in text.

## For Beginners

Stereotypes are oversimplified beliefs about groups of people
(e.g., "women are emotional" or "elderly people are slow with technology"). This module
detects when AI output reinforces such stereotypes, even when the language is subtle.

## How It Works

Identifies text that reinforces harmful stereotypes by detecting co-occurrence of demographic
group terms with stereotype-associated attributes. Uses a curated database of known stereotype
patterns across gender, racial, age, and other demographic dimensions.

**References:**

- StereoSet: Measuring stereotypical bias in pretrained language models (ACL 2021)
- CrowS-Pairs: Challenging dataset for measuring social biases (EMNLP 2020)
- BEATS: Comprehensive bias evaluation test suite for LLMs (2025)
- SB-Bench: Stereotype bias benchmark for multimodal models (2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StereotypeDetector(Double)` | Initializes a new stereotype detector. |

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

