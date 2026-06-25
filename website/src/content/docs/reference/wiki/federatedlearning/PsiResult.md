---
title: "PsiResult"
description: "Contains the results of a Private Set Intersection computation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.PSI`

Contains the results of a Private Set Intersection computation.

## For Beginners

After running PSI, this object tells you:

## How It Works

The alignment mappings are critical for vertical FL training: they tell each party
which of their local data rows correspond to the shared entities so that features from
different parties can be correctly paired during training.

## Properties

| Property | Summary |
|:-----|:--------|
| `ExecutionTime` | Gets or sets the total time taken to execute the PSI protocol. |
| `FuzzyMatchConfidences` | Gets or sets confidence scores for fuzzy-matched pairs, keyed by shared index. |
| `IntersectionIds` | Gets or sets the intersecting entity IDs found by the PSI protocol. |
| `IntersectionSize` | Gets or sets the number of intersecting elements. |
| `IsFuzzyMatch` | Gets or sets whether the result is from a fuzzy (approximate) match. |
| `LocalOverlapRatio` | Gets or sets the fraction of the initiating party's IDs that were found in the intersection. |
| `LocalToSharedIndexMap` | Gets or sets the mapping from local row indices to shared alignment indices for the initiating party. |
| `ProtocolUsed` | Gets or sets the PSI protocol that was used. |
| `RemoteOverlapRatio` | Gets or sets the fraction of the remote party's IDs that were found in the intersection. |
| `RemoteToSharedIndexMap` | Gets or sets the mapping from local row indices to shared alignment indices for the remote party. |

