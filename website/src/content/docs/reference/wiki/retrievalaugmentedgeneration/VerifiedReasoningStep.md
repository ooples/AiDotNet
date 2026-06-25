---
title: "VerifiedReasoningStep<T>"
description: "Represents a reasoning step with verification information."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Models`

Represents a reasoning step with verification information.

## Properties

| Property | Summary |
|:-----|:--------|
| `CritiqueFeedback` | Critique feedback from the critic model. |
| `IsVerified` | Whether this step passed verification. |
| `OriginalStatement` | Original statement before refinement (if any). |
| `RefinementAttempts` | Number of refinement attempts for this step. |
| `Statement` | The reasoning statement. |
| `SupportingDocuments` | Documents supporting this reasoning step. |
| `VerificationScore` | Verification score (0-1, higher is better). |

