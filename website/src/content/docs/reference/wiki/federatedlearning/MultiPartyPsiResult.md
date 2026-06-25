---
title: "MultiPartyPsiResult"
description: "Contains results of a multi-party PSI computation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.PSI`

Contains results of a multi-party PSI computation.

## For Beginners

Similar to `PsiResult` but with alignment mappings
for every participating party, not just two.

## Properties

| Property | Summary |
|:-----|:--------|
| `ExecutionTime` | Gets or sets the total execution time. |
| `IntersectionIds` | Gets or sets the intersecting entity IDs found across all parties. |
| `IntersectionSize` | Gets or sets the number of intersecting elements. |
| `NumberOfParties` | Gets or sets the number of parties that participated. |
| `PartyAlignmentMappings` | Gets or sets per-party alignment mappings (localIndex -> sharedIndex). |

