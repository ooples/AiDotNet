---
title: "CritiqueResult<T>"
description: "Result of critiquing a reasoning step or chain."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Result of critiquing a reasoning step or chain.

## For Beginners

This is like getting your homework back with a grade and comments.
The score tells you how well you did, and the feedback explains what was good or bad.

## Properties

| Property | Summary |
|:-----|:--------|
| `Feedback` | Detailed feedback explaining the score. |
| `PassesThreshold` | Whether this reasoning step/chain passes the minimum quality threshold. |
| `Score` | Quality score for the reasoning (typically 0.0 to 1.0). |
| `Strengths` | Specific strengths identified in the reasoning. |
| `Suggestions` | Suggestions for how to improve this reasoning. |
| `Weaknesses` | Specific weaknesses or areas for improvement. |

