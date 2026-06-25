---
title: "LinearFeatureMapper<T>"
description: "Implements a simple linear projection for mapping features between domains."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TransferLearning.FeatureMapping`

Implements a simple linear projection for mapping features between domains.

## For Beginners

Linear feature mapping is the simplest approach to translating between domains.
It uses matrix multiplication to transform features, similar to how you might resize an image
using simple scaling. While not as sophisticated as other methods, it's fast and works well
when domains are reasonably similar.

## How It Works

Think of it like using a simple multiplication factor to convert between units
(like converting feet to meters). It's not perfect for all situations, but it's
a good starting point.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LinearFeatureMapper` | Initializes a new instance of the LinearFeatureMapper class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsTrained` | Indicates whether the mapper has been trained. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CenterData(Matrix<>,Vector<>)` | Centers the data by subtracting the mean from each column. |
| `ComputeMean(Matrix<>)` | Computes the mean of each feature column in the data matrix. |
| `ComputeProjectionMatrix(Matrix<>,Int32,Int32)` | Computes a projection matrix using a simplified approach. |
| `ComputeReconstructionConfidence(Matrix<>,Matrix<>)` | Computes the reconstruction confidence based on how well we can round-trip the data. |
| `GetMappingConfidence` | Gets the confidence score for the mapping quality. |
| `MapToSource(Matrix<>,Int32)` | Maps features from target domain back to source domain. |
| `MapToTarget(Matrix<>,Int32)` | Maps features from source domain to target domain. |
| `OrthonormalizeColumns(Matrix<>)` | Orthonormalizes the columns of a matrix using the Gram-Schmidt process. |
| `ScaleVector(Vector<>,)` | Scales a vector by a scalar value. |
| `SubtractScaled(Vector<>,Vector<>,)` | Subtracts a scaled vector from another vector. |
| `Train(Matrix<>,Matrix<>)` | Trains the linear feature mapper using Principal Component Analysis (PCA)-like approach. |

