---
title: "IFeatureMapper<T>"
description: "Defines the interface for mapping features from a source domain to a target domain."
section: "API Reference"
---

`Interfaces` · `AiDotNet.TransferLearning.FeatureMapping`

Defines the interface for mapping features from a source domain to a target domain.

## For Beginners

A feature mapper is like a translator between two different languages.
When you have data from one domain (source) and want to use it in another domain (target),
the feature mapper transforms the data so it makes sense in the new context.

## How It Works

For example, if you trained a model on images (which might have thousands of features)
and want to use that knowledge for text (which has different features), a feature mapper
helps bridge that gap by finding a common representation.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsTrained` | Determines if the mapper has been trained and is ready to use. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetMappingConfidence` | Gets the confidence score for the mapping quality. |
| `MapToSource(Matrix<>,Int32)` | Maps features from the target domain back to the source domain. |
| `MapToTarget(Matrix<>,Int32)` | Maps features from the source domain to the target domain. |
| `Train(Matrix<>,Matrix<>)` | Trains the feature mapper on source and target data. |

