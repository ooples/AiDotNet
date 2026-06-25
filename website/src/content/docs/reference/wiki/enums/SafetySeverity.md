---
title: "SafetySeverity"
description: "Indicates the severity level of a safety finding."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Indicates the severity level of a safety finding.

## For Beginners

Not all safety issues are equally serious. A mild profanity
is different from CSAM or weapons instructions. Severity levels help you set
different thresholds and actions for different levels of risk.

## How It Works

Severity levels help prioritize safety findings and determine the appropriate response.
They follow standard security severity conventions.

## Fields

| Field | Summary |
|:-----|:--------|
| `Critical` | Critical severity — content involves illegal activity, CSAM, or imminent danger. |
| `High` | High severity — content is clearly harmful or dangerous. |
| `Info` | Informational finding with no safety risk. |
| `Low` | Low severity — content is borderline or slightly inappropriate. |
| `Medium` | Medium severity — content contains moderate safety concerns. |

