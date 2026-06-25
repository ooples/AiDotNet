---
title: "Transfer Learning"
description: "All 9 public types in the AiDotNet.transferlearning namespace, organized by kind."
section: "API Reference"
---

**9** public types in this namespace, organized by kind.

## Models & Types (6)

| Type | Summary |
|:-----|:--------|
| [`CORALDomainAdapter<T>`](/docs/reference/wiki/transferlearning/coraldomainadapter/) | Implements domain adaptation using CORrelation ALignment (CORAL). |
| [`LinearFeatureMapper<T>`](/docs/reference/wiki/transferlearning/linearfeaturemapper/) | Implements a simple linear projection for mapping features between domains. |
| [`MMDDomainAdapter<T>`](/docs/reference/wiki/transferlearning/mmddomainadapter/) | Implements domain adaptation using Maximum Mean Discrepancy (MMD). |
| [`MappedRandomForestModel<T>`](/docs/reference/wiki/transferlearning/mappedrandomforestmodel/) | Wrapper model that applies feature mapping before prediction. |
| [`TransferNeuralNetwork<T>`](/docs/reference/wiki/transferlearning/transferneuralnetwork/) | Implements transfer learning for Neural Network models. |
| [`TransferRandomForest<T>`](/docs/reference/wiki/transferlearning/transferrandomforest/) | Implements transfer learning for Random Forest models. |

## Base Classes (1)

| Type | Summary |
|:-----|:--------|
| [`TransferLearningBase<T, TInput, TOutput>`](/docs/reference/wiki/transferlearning/transferlearningbase/) |  |

## Interfaces (2)

| Type | Summary |
|:-----|:--------|
| [`IDomainAdapter<T>`](/docs/reference/wiki/transferlearning/idomainadapter/) | Defines the interface for adapting models to reduce distribution shift between source and target domains. |
| [`IFeatureMapper<T>`](/docs/reference/wiki/transferlearning/ifeaturemapper/) | Defines the interface for mapping features from a source domain to a target domain. |

