---
title: "AnchorExplanation<T>"
description: "Represents an anchor explanation providing rule-based interpretations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability`

Represents an anchor explanation providing rule-based interpretations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AnchorExplanation` | Initializes a new instance of the AnchorExplanation class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AnchorFeatures` | Gets or sets the features involved in the anchor. |
| `AnchorRules` | Gets or sets the anchor rules (feature indices and their conditions). |
| `Coverage` | Gets or sets the coverage of the anchor (fraction of instances covered). |
| `Description` | Gets or sets a human-readable description of the anchor rules. |
| `Precision` | Gets or sets the precision of the anchor (how often the anchor holds). |
| `Threshold` | Gets or sets the threshold used for anchor construction. |

