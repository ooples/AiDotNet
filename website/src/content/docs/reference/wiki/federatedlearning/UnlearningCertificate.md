---
title: "UnlearningCertificate"
description: "Certificate proving that a client's data has been unlearned from the federated model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Unlearning`

Certificate proving that a client's data has been unlearned from the federated model.

## For Beginners

When a client requests data removal, the system needs to prove it
actually happened. This certificate contains verifiable metrics showing the client's contribution
has been removed. Think of it as a "receipt of forgetting" — proof for GDPR auditors that the
right to be forgotten was honored.

## How It Works

**Key metrics:**

## Properties

| Property | Summary |
|:-----|:--------|
| `ClientRoundsParticipated` | Gets or sets the number of training rounds the target client participated in. |
| `MembershipInferenceScore` | Gets or sets the membership inference attack score after unlearning. |
| `MethodUsed` | Gets or sets the unlearning method used. |
| `ModelDivergence` | Gets or sets the L2 distance between the original and unlearned model parameters. |
| `PostUnlearningModelHash` | Gets or sets a hash of the model state after unlearning. |
| `PreUnlearningModelHash` | Gets or sets a hash of the model state before unlearning (for audit trail). |
| `RetainedAccuracy` | Gets or sets the retained accuracy on non-target clients after unlearning. |
| `Summary` | Gets or sets an optional human-readable summary of the unlearning result. |
| `TargetClientId` | Gets or sets the client ID whose data was unlearned. |
| `Timestamp` | Gets or sets the UTC timestamp when unlearning was performed. |
| `UnlearningTimeMs` | Gets or sets the wall-clock time in milliseconds the unlearning took. |
| `Verified` | Gets or sets whether the unlearning was verified as correct. |

