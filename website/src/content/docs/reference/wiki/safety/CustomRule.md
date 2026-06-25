---
title: "CustomRule"
description: "A custom safety rule that evaluates text and optionally returns a finding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Guardrails`

A custom safety rule that evaluates text and optionally returns a finding.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CustomRule(String,Func<String,SafetyFinding>)` | Creates a new custom rule. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this custom rule. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(String)` | Evaluates the text against this rule. |

