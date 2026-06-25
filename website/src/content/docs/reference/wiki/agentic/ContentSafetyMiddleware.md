---
title: "ContentSafetyMiddleware"
description: "A guardrail `IChatMiddleware` that screens the user input before the model is called and/or the model's response after, using an `IContentModerator`."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Pipeline`

A guardrail `IChatMiddleware` that screens the user input before the model is called and/or the
model's response after, using an `IContentModerator`. On a violation it short-circuits with a
refusal (finish reason `ContentFilter`) or throws, per configuration.

## For Beginners

A bouncer for the model. It can refuse risky requests before they reach the
model and catch unsafe answers before they reach the user — returning a polite refusal (or raising an
error if you prefer to stop hard).

## How It Works

As middleware it composes with any `IChatClient` and stacks with logging/telemetry. Input
screening blocks before incurring a model call; output screening protects against unsafe completions even
from a benign prompt.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ContentSafetyMiddleware(IContentModerator,ContentSafetyOptions)` | Initializes a new content-safety middleware. |

## Methods

| Method | Summary |
|:-----|:--------|
| `InvokeAsync(ChatRequestContext,ChatPipelineDelegate,CancellationToken)` |  |

