---
title: "ReasoningStep<T>"
description: "Represents a single step in a reasoning chain, capturing the thought process and evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.Models`

Represents a single step in a reasoning chain, capturing the thought process and evaluation.

## For Beginners

Think of a reasoning step like a single line in showing your work on a math problem.
When you solve "What is 15% of 240?", your steps might be:

- Step 1: "Convert 15% to decimal: 15/100 = 0.15"
- Step 2: "Multiply by 240: 0.15 × 240 = 36"
- Step 3: "Therefore, the answer is 36"

Each ReasoningStep captures:

- What you thought (the reasoning text)
- How confident you are (the score)
- Whether this step was verified/checked
- Any feedback or corrections made

## How It Works

**Example Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ReasoningStep` | Initializes a new instance of the `ReasoningStep` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Content` | The actual reasoning content or thought for this step. |
| `CreatedAt` | Timestamp when this step was created. |
| `CriticFeedback` | Feedback from critic model if verification was performed. |
| `ExternalVerificationResult` | Result from external verification tool if applicable. |
| `IsVerified` | Whether this step has been verified by a critic model or external tool. |
| `Metadata` | Additional metadata or context specific to this step. |
| `OriginalContent` | Original content before any refinement (if the step was refined). |
| `RefinementCount` | Number of times this step was refined/corrected. |
| `Score` | Confidence or quality score for this step (typically 0.0 to 1.0). |
| `StepNumber` | The sequential number of this step in the reasoning chain (starting from 1). |
| `VerificationMethod` | Tool or method used to verify this step externally (e.g., "Calculator", "CodeExecution"). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Returns a string representation of this reasoning step. |

