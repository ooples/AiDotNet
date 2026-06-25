---
title: "EasyEnsembleClassifier<T>"
description: "Easy Ensemble Classifier for extremely imbalanced datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.ImbalancedEnsemble`

Easy Ensemble Classifier for extremely imbalanced datasets.

## For Beginners

Easy Ensemble is designed for highly imbalanced data. It creates
multiple balanced subsets by undersampling the majority class, trains a base classifier
(typically AdaBoost) on each subset, and combines their predictions.

## How It Works

**How it works:**

- Keep all minority class samples
- For each subset: randomly undersample majority to match minority
- Train an AdaBoost classifier on each balanced subset
- Combine predictions using majority voting or averaging

**Key advantages:**

- **Handles extreme imbalance:** Works well even with 100:1 or 1000:1 ratios
- **Uses all majority data:** Through multiple subsets, all majority samples contribute
- **Reduces variance:** Ensemble approach gives stable predictions

**When to use:**

- Extremely imbalanced datasets (more than 10:1 ratio)
- When you need high recall for the minority class
- When losing some majority class accuracy is acceptable

**References:**

- Liu, X.Y., Wu, J., & Zhou, Z.H. (2009). "Exploratory Undersampling for Class-Imbalance Learning"

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EasyEnsembleClassifier(Int32,Int32,Int32,Double,String,Boolean,Nullable<Int32>)` | Initializes a new instance of EasyEnsembleClassifier. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies gradients (not applicable for ensemble models). |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` | Computes gradients (not applicable for ensemble models). |
| `CreateBalancedSubset(Dictionary<Int32,List<Int32>>,Int32,Int32)` | Creates a balanced subset by undersampling majority class. |
| `CreateNewInstance` | Creates a new instance of this model type. |
| `Deserialize(Byte[])` | Deserializes the trained model state including all AdaBoost sub-classifiers. |
| `GetFeatureImportance` | Gets feature importance based on weak learner usage. |
| `GetParameters` | Gets the model parameters. |
| `Predict(Matrix<>)` | Predicts class labels for the given input data. |
| `PredictAdaBoost(EasyEnsembleClassifier<>.AdaBoostSubClassifier,Matrix<>,Int32)` | Gets the AdaBoost class prediction for a sample. |
| `PredictAdaBoostScore(EasyEnsembleClassifier<>.AdaBoostSubClassifier,Matrix<>,Int32)` | Gets the raw AdaBoost score for a sample. |
| `PredictWeakLearner(EasyEnsembleClassifier<>.WeakLearner,Matrix<>,Int32)` | Gets prediction from a weak learner. |
| `Serialize` | Serializes the trained model state including all AdaBoost sub-classifiers. |
| `SetParameters(Vector<>)` | Sets the model parameters. |
| `Train(Matrix<>,Vector<>)` | Gets the model type. |
| `TrainAdaBoost(Matrix<>,Vector<>)` | Trains an AdaBoost classifier. |
| `TrainWeakLearnerFast(Double[][],Int32[],Double[],Int32)` | Trains a weak learner (decision stump or shallow tree). |
| `WithParameters(Vector<>)` | Creates a new instance with the specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_learningRate` | Learning rate for AdaBoost. |
| `_maxDepth` | Maximum depth for weak learners in AdaBoost. |
| `_nEstimatorsPerSubset` | Number of AdaBoost estimators per subset. |
| `_nSubsets` | Number of subset classifiers. |
| `_random` | Random number generator. |
| `_samplingStrategy` | Sampling strategy for undersampling. |
| `_softVoting` | Whether to use soft voting (probability averaging) vs hard voting. |
| `_subClassifiers` | The ensemble of AdaBoost classifiers. |

