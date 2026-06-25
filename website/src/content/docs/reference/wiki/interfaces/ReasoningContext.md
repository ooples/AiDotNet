---
title: "ReasoningContext"
description: "Context information for critiquing reasoning steps."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Context information for critiquing reasoning steps.

## For Beginners

This provides the critic with background information needed
to properly evaluate a reasoning step, like giving a teacher the full assignment when
grading one answer.

## Properties

| Property | Summary |
|:-----|:--------|
| `Domain` | Domain or subject area (e.g., "mathematics", "code", "science"). |
| `PreviousSteps` | Previous reasoning steps that provide context. |
| `Query` | The original query or problem being solved. |
| `SupportingEvidence` | Supporting evidence or documents if available. |

