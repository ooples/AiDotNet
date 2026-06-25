---
title: "GuardrailRule"
description: "Defines a declarative guardrail rule with a condition expression and corresponding action."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Guardrails`

Defines a declarative guardrail rule with a condition expression and corresponding action.

## For Beginners

A rule is like an if-then statement for safety. The condition says
"if the text contains X" and the action says "then do Y (block, warn, log, or modify)."
You can build complex safety policies by combining multiple rules.

## How It Works

A guardrail rule pairs a condition (what to detect) with an action (what to do about it).
Rules can check for regex patterns, keyword lists, length limits, or custom predicates.
Unlike `CustomRule` which uses raw delegates, GuardrailRule uses a structured
format that enables serialization, logging, and composition.

**References:**

- Guardrails AI: Structured validators (2024)
- NeMo Guardrails: Colang rule language (NVIDIA, 2024)

## Properties

| Property | Summary |
|:-----|:--------|
| `Action` | Gets or sets the action to take when the condition is met. |
| `CaseInsensitive` | Gets or sets whether pattern matching is case-insensitive. |
| `Category` | Gets or sets the safety category to assign when the rule triggers. |
| `ConditionType` | Gets or sets the type of condition this rule uses. |
| `CustomPredicate` | Gets or sets an optional custom predicate for `Custom` rules. |
| `Description` | Gets or sets the human-readable description of what this rule checks. |
| `Direction` | Gets or sets the direction this rule applies to. |
| `Name` | Gets or sets the unique name for this rule. |
| `Patterns` | Gets or sets the patterns or keywords to check for (interpretation depends on ConditionType). |
| `Severity` | Gets or sets the severity to assign when the rule triggers. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(String)` | Evaluates this rule against the given text. |

