---
title: "FederatedPersonalizationOptions"
description: "Configuration options for personalized federated learning (PFL)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for personalized federated learning (PFL).

## How It Works

**For Beginners:** Personalization means each client can end up with a model that works better for its own data,
while still learning shared knowledge from other clients.

This options class controls which personalization algorithm is used (FedPer, FedRep, Ditto, pFedMe, clustered, etc.)
and the key hyperparameters for those algorithms.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClusterCount` | Gets or sets the number of clusters used for clustered personalization. |
| `DittoLambda` | Gets or sets the Ditto regularization strength (lambda). |
| `Enabled` | Gets or sets whether personalization is enabled. |
| `FedAGHNSharedDimension` | Gets or sets the shared dimension for adaptive gradient heterogeneous networks. |
| `FedBABUHeadFraction` | Gets or sets the fraction of parameters considered "head" in FedBABU. |
| `FedBABULocalFineTuneEpochs` | Gets or sets the number of local fine-tune epochs after receiving the global model (FedBABU). |
| `FedCPNumExperts` | Gets or sets the number of conditional computation experts in FedCP. |
| `FedPACCalibrationWeight` | Gets or sets the calibration weight for FedPAC post-aggregation calibration. |
| `FedPACSimilarityThreshold` | Gets or sets the cosine similarity threshold for FedPAC prototype alignment. |
| `FedRoDMixingAlpha` | Gets or sets the mixing alpha between generic and personalized classifier in FedRoD. |
| `FedSelectMaskRegularization` | Gets or sets the mask regularization strength for FedSelect sparsity. |
| `FedSelectMaskThreshold` | Gets or sets the binary mask threshold for FedSelect parameter selection. |
| `KNNPerK` | Gets or sets the number of nearest neighbors (k) for kNN-Per inference. |
| `KNNPerLambda` | Gets or sets the interpolation strength between model predictions and kNN predictions. |
| `LocalAdaptationEpochs` | Gets or sets the number of extra local adaptation epochs applied after receiving the aggregated global model. |
| `PFedGateInitValue` | Gets or sets the initial gate value for pFedGate (0 = fully global, 1 = fully local). |
| `PFedGateLearningRate` | Gets or sets the gate learning rate for pFedGate. |
| `PFedMeInnerSteps` | Gets or sets the number of inner proximal steps for pFedMe (K). |
| `PFedMeMu` | Gets or sets the pFedMe proximal strength (mu). |
| `PersonalizedParameterFraction` | Gets or sets the fraction of parameters treated as "personalized" (not aggregated globally). |
| `Strategy` | Gets or sets the personalization strategy. |

