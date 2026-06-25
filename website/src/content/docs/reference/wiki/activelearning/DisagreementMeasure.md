---
title: "DisagreementMeasure"
description: "Methods for measuring disagreement in a committee of models."
section: "API Reference"
---

`Enums` · `AiDotNet.ActiveLearning.Config`

Methods for measuring disagreement in a committee of models.

## Fields

| Field | Summary |
|:-----|:--------|
| `ConsensusEntropy` | Entropy of the averaged probability predictions. |
| `KullbackLeiblerDivergence` | Average KL divergence between individual and consensus predictions. |
| `MaxDisagreement` | Maximum disagreement between any two committee members. |
| `VoteEntropy` | Entropy of the vote distribution across committee members. |

