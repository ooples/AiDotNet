---
title: "TransferLearningBase<T, TInput, TOutput>"
description: "TransferLearningBase<T, TInput, TOutput> — Base Classes in AiDotNet.TransferLearning.Algorithms."
section: "API Reference"
---

`Base Classes` · `AiDotNet.TransferLearning.Algorithms`

_No summary documentation available yet._

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TransferLearningBase` | Initializes a new instance of the TransferLearningBase class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeCentroid()` | Computes the centroid (mean) of a data matrix. |
| `ComputeEuclideanDistance(Vector<>,Vector<>)` | Computes the Euclidean distance between two vectors. |
| `ComputeTransferConfidence(,)` | Computes a confidence score for transfer learning success. |
| `GetRowAsVector(,Int32)` | Gets a single row from TInput as a Vector. |
| `RequiresCrossDomainTransfer(IFullModel<,,>,)` | Evaluates if cross-domain transfer is necessary based on feature dimensions. |
| `SelectRelevantSourceSamples(,,Double)` | Selects the most relevant samples from source domain for transfer. |
| `SetDomainAdapter(IDomainAdapter<>)` | Sets the domain adapter to use for reducing distribution shift. |
| `SetFeatureMapper(IFeatureMapper<>)` | Sets the feature mapper to use for cross-domain transfer. |
| `TransferCrossDomain(IFullModel<,,>,,)` | Transfers knowledge from a source model to a target domain (different feature space). |
| `TransferSameDomain(IFullModel<,,>,,)` | Transfers knowledge from a source model to a target domain (same feature space). |

