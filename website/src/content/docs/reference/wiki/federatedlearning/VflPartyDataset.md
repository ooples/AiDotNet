---
title: "VflPartyDataset<T>"
description: "Contains data for a single party in a benchmark dataset."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Vertical`

Contains data for a single party in a benchmark dataset.

## Properties

| Property | Summary |
|:-----|:--------|
| `EntityIds` | Gets or sets the entity IDs for this party. |
| `FeatureColumnIndices` | Gets or sets which columns from the full dataset this party holds. |
| `Features` | Gets or sets the party's feature data [numEntities, numFeatures]. |
| `IsLabelHolder` | Gets or sets whether this party is the label holder. |
| `Labels` | Gets or sets the labels (only for label holder). |
| `PartyId` | Gets or sets the party identifier. |

