---
title: "ContentSafetyOptions"
description: "Settings for `ContentSafetyMiddleware`: which sides to screen, what to say when blocking, and whether a violation throws or returns a refusal."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Agentic.Pipeline`

Settings for `ContentSafetyMiddleware`: which sides to screen, what to say when blocking, and
whether a violation throws or returns a refusal.

## For Beginners

The dials for the safety filter — check the user's message, the model's reply,
or both; the message to show when something is blocked; and whether a block is a polite refusal (default)
or a hard error.

## Properties

| Property | Summary |
|:-----|:--------|
| `CheckInput` | Gets or sets whether to screen the user input before calling the model. |
| `CheckOutput` | Gets or sets whether to screen the model's response. |
| `RefusalMessage` | Gets or sets the assistant message returned when content is blocked and `ThrowOnViolation` is `false`. |
| `ThrowOnViolation` | Gets or sets whether a violation throws a `ContentSafetyException` (`true`) instead of returning a refusal response (`false`, the default). |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultRefusalMessage` | The default message returned when content is blocked. |

