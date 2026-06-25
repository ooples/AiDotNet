---
title: "ContentSafetyException"
description: "Thrown by `ContentSafetyMiddleware` when content is blocked and the middleware is configured to fail hard (`ThrowOnViolation`) rather than return a refusal."
section: "API Reference"
---

`Exceptions` · `AiDotNet.Agentic.Pipeline`

Thrown by `ContentSafetyMiddleware` when content is blocked and the middleware is configured to
fail hard (`ThrowOnViolation`) rather than return a refusal.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ContentSafetyException(String)` | Initializes a new exception describing why content was blocked. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Reason` | Gets the moderation reason the content was blocked. |

