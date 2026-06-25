---
title: "SafetyValidationResult<T>"
description: "Result of safety validation for an input."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Result of safety validation for an input.

## Properties

| Property | Summary |
|:-----|:--------|
| `DetectedHarmCategories` | Gets or sets harmful content categories detected. |
| `IsValid` | Gets or sets whether the input passed validation. |
| `Issues` | Gets or sets the list of validation issues found. |
| `JailbreakDetected` | Gets or sets whether a jailbreak attempt was detected. |
| `SafetyScore` | Gets or sets the safety score (0-1, higher is safer). |
| `SanitizedInput` | Gets or sets the sanitized/cleaned input (if applicable). |
| `ValidationDetails` | Gets or sets additional validation details. |

