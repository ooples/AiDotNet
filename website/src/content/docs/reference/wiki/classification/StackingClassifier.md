---
title: "StackingClassifier<T>"
description: "Stacking classifier that uses predictions from base classifiers as features for a meta-classifier."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Meta`

Stacking classifier that uses predictions from base classifiers as features for a meta-classifier.

## For Beginners

Stacking is a sophisticated ensemble method:

1. Train multiple base classifiers
2. Get predictions from each on training data (using cross-validation)
3. Use these predictions as features for a "meta" classifier
4. Train the meta-classifier on these stacked predictions

For prediction:

1. Get predictions from all base classifiers
2. Stack them as features
3. Feed to meta-classifier for final prediction

Benefits:

- Can combine very different types of classifiers
- Often achieves better accuracy than individual classifiers
- The meta-classifier learns which base classifiers to trust

Considerations:

- More complex to implement
- Risk of overfitting if not using cross-validation
- Computationally expensive

## How It Works

Stacking trains multiple base classifiers and then uses their predictions
as features to train a final meta-classifier.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StackingClassifier` | Initializes a new instance with default settings using Gaussian Naive Bayes estimators. |
| `StackingClassifier(IEnumerable<IClassifier<>>,Func<IClassifier<>>,StackingClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the StackingClassifier class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Options` | Gets the stacking-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateCrossValidatedMetaFeatures(Matrix<>,Vector<>,Matrix<>)` | Creates meta-features using cross-validation. |
| `CreateNewInstance` |  |
| `CreatePredictionMetaFeatures(Matrix<>)` | Creates meta-features for prediction. |
| `CreateSimpleMetaFeatures(Matrix<>,Vector<>,Matrix<>)` | Creates meta-features without cross-validation. |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `Predict(Matrix<>)` |  |
| `PredictLogProbabilities(Matrix<>)` |  |
| `PredictProbabilities(Matrix<>)` |  |
| `Serialize` |  |
| `Train(Matrix<>,Vector<>)` | Returns the model type identifier for this classifier. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_estimators` | The base estimators. |
| `_finalEstimator` | The meta-classifier (final estimator). |
| `_finalEstimatorFactory` | Factory for creating final estimator. |

