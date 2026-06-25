---
title: "IContentModerator"
description: "Checks whether a piece of text is safe/allowed — the seam guardrails use to screen agent inputs and outputs."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Agentic.Pipeline`

Checks whether a piece of text is safe/allowed — the seam guardrails use to screen agent inputs and
outputs. Implementations range from simple deny-lists to PII/jailbreak detectors or an
`src/Safety`-backed classifier.

## For Beginners

A safety checker. Give it some text and it says "fine" or "blocked, because…".
The guardrail middleware calls it on what users send in and on what the model sends back.

## Methods

| Method | Summary |
|:-----|:--------|
| `CheckAsync(String,CancellationToken)` | Moderates a piece of content. |

