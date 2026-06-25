---
title: "MetaClassifierBase<T>"
description: "Base class for meta classifiers that wrap other classifiers."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Classification.Meta`

Base class for meta classifiers that wrap other classifiers.

## For Beginners

Meta classifiers are "classifiers about classifiers":

Examples:

- OneVsRest: Trains one classifier per class
- OneVsOne: Trains one classifier per pair of classes
- Voting: Combines predictions from multiple classifiers
- Stacking: Uses classifier outputs as features for another classifier

They extend the capabilities of base classifiers.

## How It Works

Meta classifiers use other classifiers as base estimators to provide
enhanced functionality like multi-class support, multi-label support,
or ensemble voting.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MetaClassifierBase(MetaClassifierOptions<>,Func<IClassifier<>>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the MetaClassifierBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EstimatorFactory` | The base estimator factory function. |
| `Options` | Gets the meta classifier specific options. |
| `SupportsParameterInitialization` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` |  |
| `CreateBaseEstimator` | Creates a new base estimator instance. |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |
| `WithParameters(Vector<>)` |  |

