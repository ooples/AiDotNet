---
title: "SafetyViolationException"
description: "Exception thrown when content fails a safety check and the configuration requires throwing."
section: "API Reference"
---

`Exceptions` · `AiDotNet.Safety`

Exception thrown when content fails a safety check and the configuration requires throwing.

## For Beginners

When the safety system finds dangerous content and is configured
to throw exceptions, this is the exception you'll catch. It contains the full
`SafetyReport` with details about what was found.

## How It Works

This exception is thrown when `ThrowOnUnsafeInput` or
`ThrowOnUnsafeOutput` is true and the safety pipeline
detects content that should be blocked.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SafetyViolationException(SafetyReport,Boolean)` | Initializes a new instance of the `SafetyViolationException` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsInputViolation` | Gets whether this violation was on input (true) or output (false). |
| `Report` | Gets the safety report containing all findings. |

