---
title: "TransferRandomForest<T>"
description: "Implements transfer learning for Random Forest models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TransferLearning.Algorithms`

Implements transfer learning for Random Forest models.

## For Beginners

This class enables Random Forest models to transfer knowledge
from one domain to another. Random Forests are ensembles of decision trees, and this
class can adapt them when the source and target domains have different feature spaces.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TransferRandomForest` | Initializes a new instance with default settings. |
| `TransferRandomForest(RandomForestRegressionOptions,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the TransferRandomForest class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CombineLabels(Vector<>,Vector<>,Double)` | Combines pseudo-labels from source model with true target labels. |
| `Transfer(IFullModel<,Matrix<>,Vector<>>,Matrix<>,Matrix<>,Vector<>)` | Transfers a Random Forest model to a target domain with proper source data. |
| `TransferCrossDomain(IFullModel<,Matrix<>,Vector<>>,Matrix<>,Vector<>)` | Transfers a Random Forest model to a target domain with a different feature space. |
| `TransferSameDomain(IFullModel<,Matrix<>,Vector<>>,Matrix<>,Vector<>)` | Transfers a Random Forest model to a target domain with the same feature space. |

