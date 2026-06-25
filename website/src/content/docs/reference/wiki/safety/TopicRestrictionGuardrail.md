---
title: "TopicRestrictionGuardrail<T>"
description: "Guardrail that restricts conversation to approved topics using keyword and pattern matching."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Guardrails`

Guardrail that restricts conversation to approved topics using keyword and pattern matching.

## For Beginners

This guardrail lets you define specific topics that the AI should
not discuss. For example, if you're building a cooking assistant, you might restrict
topics like "politics", "religion", or "medical advice". If a user asks about a restricted
topic, the system will block or warn.

## How It Works

Blocks or warns when input contains references to restricted topics. Uses both exact
keyword matching and configurable regex patterns. Topics are case-insensitive.

**References:**

- NeMo Guardrails: Topic control through conversational rails (NVIDIA, 2024)
- Guardrails AI: Semantic topic filtering (2024)
- Constitutional AI: Principle-based output steering (Anthropic, 2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TopicRestrictionGuardrail(String[],SafetyAction)` | Initializes a new topic restriction guardrail. |

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

