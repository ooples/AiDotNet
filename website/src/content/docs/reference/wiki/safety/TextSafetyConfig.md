---
title: "TextSafetyConfig"
description: "Configuration for text safety modules."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Safety`

Configuration for text safety modules.

## For Beginners

These settings control how text content is checked for safety issues.
Enable the checks you need — toxicity catches offensive language, PII detection finds
personal information, jailbreak detection catches prompt manipulation attempts.

## How It Works

Controls which text safety checks are enabled and their sensitivity thresholds.

## Properties

| Property | Summary |
|:-----|:--------|
| `CopyrightDetection` | Gets or sets whether copyright/memorization detection is enabled. |
| `HallucinationDetection` | Gets or sets whether hallucination detection is enabled. |
| `JailbreakDetection` | Gets or sets whether jailbreak/prompt injection detection is enabled. |
| `JailbreakSensitivity` | Gets or sets the jailbreak detection sensitivity (0-1). |
| `Languages` | Gets or sets the languages to support for text safety checks. |
| `MaxInputLength` | Gets or sets the maximum input text length. |
| `PIIDetection` | Gets or sets whether PII (personally identifiable information) detection is enabled. |
| `ToxicityDetection` | Gets or sets whether toxicity detection is enabled. |
| `ToxicityThreshold` | Gets or sets the toxicity score threshold (0-1). |

