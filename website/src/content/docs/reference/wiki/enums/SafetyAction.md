---
title: "SafetyAction"
description: "Defines the action to take when a safety violation is detected."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the action to take when a safety violation is detected.

## For Beginners

When the safety system finds something potentially harmful,
it needs to decide what to do. These are the possible responses:

- Allow: Let it through (useful for logging-only modes)
- Log: Let it through but record it for review
- Warn: Let it through but attach a warning
- Modify: Change the content to make it safe (e.g., redact PII)
- Block: Stop the content entirely
- Quarantine: Block and flag for human review

## How It Works

Safety actions determine what happens when content fails a safety check.
Actions are ordered by severity, from most permissive to most restrictive.

## Fields

| Field | Summary |
|:-----|:--------|
| `Allow` | Allow the content through without modification. |
| `Block` | Block the content entirely and return an error or safe fallback. |
| `Log` | Allow the content but log the safety finding for later review. |
| `Modify` | Modify the content to remove or redact the unsafe portion. |
| `Quarantine` | Block the content and flag it for human review. |
| `Warn` | Allow the content but attach a warning to the result. |

