---
title: "InputGuardrail<T>"
description: "Input guardrail that validates user input before it reaches the model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Guardrails`

Input guardrail that validates user input before it reaches the model.

## For Beginners

This guardrail checks user input before it's processed by the AI.
It catches problems like inputs that are too long, empty inputs, or inputs that contain
suspicious patterns (like prompt injection attempts using special characters).

## How It Works

Enforces structural input constraints including maximum length, required format,
and basic sanity checks. Acts as the first line of defense before content-specific
safety modules run.

**References:**

- NeMo Guardrails: Input/output rails for LLM applications (NVIDIA, 2024)
- Guardrails AI: Production-grade validation framework (2024)
- LLM Guard: Input validation for production systems (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InputGuardrail(Int32,Boolean,Boolean)` | Initializes a new input guardrail. |

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

