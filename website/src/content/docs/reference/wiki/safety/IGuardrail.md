---
title: "IGuardrail<T>"
description: "Interface for guardrail modules that validate and filter input/output content."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Safety.Guardrails`

Interface for guardrail modules that validate and filter input/output content.

## For Beginners

A guardrail is like a safety fence around your AI. Input guardrails
check what goes into the AI (blocking dangerous prompts), and output guardrails check
what comes out (filtering harmful responses). You can also restrict specific topics.

## How It Works

Guardrails are pre/post-processing safety checks that validate content before and after
model processing. They enforce length limits, topic restrictions, content policies,
and custom rules. Guardrails can block, warn, log, or modify content.

## Properties

| Property | Summary |
|:-----|:--------|
| `Direction` | Gets whether this guardrail checks input content, output content, or both. |

