---
title: "OutputGuardrail<T>"
description: "Output guardrail that validates model output before it reaches the user."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Guardrails`

Output guardrail that validates model output before it reaches the user.

## For Beginners

This guardrail checks the AI's response before you see it. It catches
problems like responses that are too short (model refusing or failing), responses that
contain the model's own system prompt (data leakage), or responses that repeat excessively.

## How It Works

Validates model responses for structural issues, refusal patterns, and output quality.
Runs after the model generates output but before it's returned to the user, catching
issues that the model may have introduced.

**References:**

- ShieldGemma: LLM-based safety content filter (Google DeepMind, 2024)
- LLaMA Guard 3: Safeguarding human-AI conversations (Meta, 2024)
- WildGuard: Open-source LLM safety moderation (Allen AI, 2024)
- Qwen3Guard: Multilingual safety evaluator (Alibaba, 2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OutputGuardrail(Int32,Int32,Double)` | Initializes a new output guardrail. |

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

