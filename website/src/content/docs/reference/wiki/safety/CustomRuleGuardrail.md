---
title: "CustomRuleGuardrail<T>"
description: "User-defined guardrail that applies custom validation rules to text content."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Guardrails`

User-defined guardrail that applies custom validation rules to text content.

## For Beginners

This guardrail lets you define your own custom safety rules.
You write a function that checks the input and returns a safety finding if something
is wrong. This is useful for application-specific rules that the built-in modules
don't cover.

## How It Works

Allows users to define arbitrary validation rules as delegates. Each rule receives
the input text and returns a SafetyFinding if the rule is violated. This provides
maximum flexibility for application-specific safety requirements.

**Example usage:**

**References:**

- Guardrails AI: Custom validators for production LLM systems (2024)
- NeMo Guardrails: Programmable rails (NVIDIA, 2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CustomRuleGuardrail(IReadOnlyList<CustomRule>)` | Initializes a new custom rule guardrail. |

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

