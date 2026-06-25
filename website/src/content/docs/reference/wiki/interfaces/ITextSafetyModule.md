---
title: "ITextSafetyModule<T>"
description: "Interface for safety modules that operate on text content."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for safety modules that operate on text content.

## For Beginners

Text safety modules check written content for problems.
Unlike the base ISafetyModule which works on numeric vectors, text safety modules
can accept raw strings directly, making them easier to use for text-based applications.

## How It Works

Text safety modules analyze string content for safety risks such as toxicity,
PII exposure, jailbreak attempts, hallucinations, and copyright violations.
They extend `ISafetyModule` with text-specific evaluation methods.

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateText(String)` | Evaluates the given text string for safety and returns any findings. |

