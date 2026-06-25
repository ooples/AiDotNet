---
title: "FederatedPersonalizationStrategy"
description: "Specifies the personalization strategy for federated learning."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the personalization strategy for federated learning.

## For Beginners

Each strategy determines how clients adapt the global model to their local data.

## Fields

| Field | Summary |
|:-----|:--------|
| `Clustered` | Clustered — cluster clients by gradient similarity, aggregate within clusters. |
| `Ditto` | Ditto — train a regularized personalized model alongside the global model. |
| `FedAGHN` | FedAGHN — adaptive gradient-based heterogeneous networks. |
| `FedBABU` | FedBABU — freeze head during FL, fine-tune body, then locally fine-tune head. |
| `FedCP` | FedCP — conditional computation policy routing inputs to model subsets. |
| `FedPAC` | FedPAC — personalization via aggregation and calibration with prototype alignment. |
| `FedPer` | FedPer — personalize the last (classification head) layers, share the body. |
| `FedRep` | FedRep — learn shared representations with personalized heads; alternating optimization. |
| `FedRoD` | FedRoD — dual classifiers: one aggregated generic + one local personalized. |
| `FedSelect` | FedSelect — learned sparse binary masks determining personalized vs shared params. |
| `KNNPer` | kNN-Per — kNN cache over global features for zero-cost personalization at inference. |
| `None` | No personalization — all parameters are aggregated globally. |
| `PFedGate` | pFedGate — gated layer-wise mixture of local and global parameters. |
| `PFedMe` | pFedMe — Moreau-envelope-based personalization with proximal local solver. |

