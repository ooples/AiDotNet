---
title: "DemographicParityChecker<T>"
description: "Checks for demographic parity violations by detecting differential treatment of demographic groups in model outputs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Fairness`

Checks for demographic parity violations by detecting differential treatment of
demographic groups in model outputs.

## For Beginners

This module checks whether AI output treats different groups of
people fairly. For example, if a model describes men positively but women negatively
when answering similar questions, that's a demographic parity violation.

## How It Works

Demographic parity requires that the probability of a positive outcome is the same
across all demographic groups. This module analyzes text for references to protected
attributes and detects sentiment/polarity differences that indicate bias.

**References:**

- BEATS: Comprehensive bias evaluation test suite for LLMs (2025)
- SB-Bench: Stereotype bias benchmark (2025)
- Demographic-targeted bias: race/ethnicity 55.6% exploitability (2025)
- Fairness metrics in machine learning survey (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DemographicParityChecker(Double,String[])` | Initializes a new demographic parity checker. |

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

