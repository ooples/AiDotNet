---
title: "VerificationResult<T>"
description: "Represents the result of external tool verification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Represents the result of external tool verification.

## For Beginners

This stores the outcome of verification - did it pass or fail,
what was the actual result, and how confident are we in the verification?

## Properties

| Property | Summary |
|:-----|:--------|
| `ActualResult` | The actual result from the external tool. |
| `Confidence` | Confidence score in the verification (0.0 to 1.0). |
| `ExpectedResult` | The expected result (from the reasoning step). |
| `Explanation` | Detailed explanation of the verification outcome. |
| `Passed` | Whether the verification passed. |
| `ToolUsed` | Name of the tool that performed the verification. |

