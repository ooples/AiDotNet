---
title: "EntityAlignmentResult"
description: "Contains the results of an entity alignment operation, including the PSI result, party sizes, and diagnostic information."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.PSI`

Contains the results of an entity alignment operation, including the PSI result,
party sizes, and diagnostic information.

## For Beginners

This wraps the raw PSI result with additional context about
the alignment operation, such as how many entities each party started with, which protocol
was used, and how long the operation took. This information is useful for logging,
monitoring, and deciding whether to proceed with VFL training.

## Properties

| Property | Summary |
|:-----|:--------|
| `AlignedEntityCount` | Gets the number of aligned entities (intersection size). |
| `IsCardinalityOnly` | Gets or sets whether only the cardinality was computed (no actual intersection IDs). |
| `IsFuzzyMatch` | Gets or sets whether fuzzy matching was used for entity alignment. |
| `LocalAlignmentRate` | Gets the fraction of the local party's entities that were aligned. |
| `LocalPartySize` | Gets or sets the number of entities in the local (initiating) party's dataset. |
| `MultiPartyResult` | Gets or sets the multi-party result when more than two parties are involved. |
| `NumberOfParties` | Gets or sets the number of parties that participated in the alignment. |
| `ProtocolUsed` | Gets or sets the name of the PSI protocol used for the alignment. |
| `PsiResult` | Gets or sets the underlying PSI result with intersection IDs and alignment mappings. |
| `RemoteAlignmentRate` | Gets the fraction of the remote party's entities that were aligned. |
| `RemotePartySize` | Gets or sets the number of entities in the remote party's dataset. |
| `TotalExecutionTime` | Gets or sets the total wall-clock time for the alignment operation. |

