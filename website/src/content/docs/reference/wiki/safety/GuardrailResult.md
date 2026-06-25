---
title: "GuardrailResult"
description: "Result from guardrail evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Guardrails`

Result from guardrail evaluation.

## For Beginners

GuardrailResult provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionTaken` | The action taken by the guardrail. |
| `ModifiedContent` | The modified content, if the guardrail applied modifications. |
| `Passed` | Whether the content passed the guardrail check. |
| `ViolatedRules` | List of violated rules. |

