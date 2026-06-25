---
title: "PredictedTriple"
description: "Represents a predicted (head, relation, tail) triple with its plausibility score."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings`

Represents a predicted (head, relation, tail) triple with its plausibility score.

## For Beginners

When the link predictor finds possible missing facts,
each prediction includes:

- The head entity, relation, and tail entity forming the predicted fact
- A score indicating how plausible the fact is
- A confidence value normalized between 0 and 1

## Properties

| Property | Summary |
|:-----|:--------|
| `Confidence` | Normalized confidence value between 0 and 1. |
| `HeadId` | The head (source) entity ID. |
| `RelationType` | The relation type connecting head to tail. |
| `Score` | Raw plausibility score from the embedding model. |
| `TailId` | The tail (target) entity ID. |

