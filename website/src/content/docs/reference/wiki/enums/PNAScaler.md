---
title: "PNAScaler"
description: "Scaler function types for Principal Neighbourhood Aggregation (PNA)."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Scaler function types for Principal Neighbourhood Aggregation (PNA).

## For Beginners

Scalers normalize aggregated features by node degree:

- **Identity**: No scaling (use raw aggregated values)
- **Amplification**: Scale up by degree/avgDegree (high-degree nodes get amplified)
- **Attenuation**: Scale down by avgDegree/degree (high-degree nodes get attenuated)

## Fields

| Field | Summary |
|:-----|:--------|
| `Amplification` | Amplification scaler - amplifies signal from high-degree nodes. |
| `Attenuation` | Attenuation scaler - attenuates signal from high-degree nodes. |
| `Identity` | Identity scaler - no scaling applied. |

