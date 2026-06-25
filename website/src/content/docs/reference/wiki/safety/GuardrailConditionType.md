---
title: "GuardrailConditionType"
description: "Specifies the type of condition a guardrail rule checks."
section: "API Reference"
---

`Enums` · `AiDotNet.Safety.Guardrails`

Specifies the type of condition a guardrail rule checks.

## Fields

| Field | Summary |
|:-----|:--------|
| `ContainsAll` | Text contains all of the specified patterns. |
| `ContainsAny` | Text contains any of the specified patterns. |
| `Custom` | Custom predicate function evaluates to true. |
| `MaxLength` | Text exceeds a maximum character length. |
| `MinLength` | Text is shorter than a minimum character length. |
| `RegexMatch` | Text matches a regular expression pattern. |

