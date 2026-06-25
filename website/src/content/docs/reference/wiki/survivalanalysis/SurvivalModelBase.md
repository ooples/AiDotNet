---
title: "SurvivalModelBase<T>"
description: "Abstract base class for survival analysis models."
section: "API Reference"
---

`Base Classes` · `AiDotNet.SurvivalAnalysis`

Abstract base class for survival analysis models.

## For Beginners

This class contains shared code that all survival models need,
so each specific model (like Cox or Kaplan-Meier) doesn't have to reimplement it.

Key shared functionality:

- Validating input data (times must be positive, events must be 0 or 1)
- Calculating the concordance index (how well the model predicts)
- Finding median survival times from survival curves
- Managing trained model state

## How It Works

This base class provides common functionality for survival models including
data validation, concordance index calculation, and baseline survival estimation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SurvivalModelBase` | Initializes a new instance of the SurvivalModelBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaselineSurvival` | Gets the baseline survival function values at event times. |
| `DefaultLossFunction` | Gets the default loss function. |
| `EventTimes` | Gets the unique event times from the training data. |
| `FeatureNames` | Gets or sets the feature names. |
| `IsTrained` | Gets whether the model is trained. |
| `NumFeatures` | Gets the number of features the model was trained on. |
| `ParameterCount` | Gets the total number of parameters in the model. |
| `SupportsParameterInitialization` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies gradients to update model parameters. |
| `CalculateConcordanceIndex(Matrix<>,Vector<>,Vector<Int32>)` | Calculates the concordance index (C-index) for model evaluation. |
| `Clone` | Creates a clone of the model. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` | Computes gradients for the given input and target. |
| `CreateNewInstance` | Creates a new instance of the same type. |
| `DeepCopy` | Creates a deep copy of the model. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `DeserializeInternalUnchecked(Byte[])` | Internal, non-virtual, no-guard deserialization used by trusted framework call sites such as `DeepCopy`. |
| `Dispose` |  |
| `Dispose(Boolean)` | Releases resources held by this survival model. |
| `EnsureFitted` | Ensures the model has been fitted before prediction. |
| `Fit(Vector<>,Vector<>,Matrix<>)` | Fits the survival model to time-to-event data (interface method). |
| `FitSurvival(Matrix<>,Vector<>,Vector<Int32>)` | Fits the survival model to time-to-event data. |
| `FitSurvivalCore(Matrix<>,Vector<>,Vector<Int32>)` | Core fitting logic to be implemented by derived classes. |
| `GetActiveFeatureIndices` | Gets the indices of features that are actively used in the model. |
| `GetBaselineSurvival(Vector<>)` | Gets the baseline survival function. |
| `GetDynamicShapeInfo` |  |
| `GetFeatureImportance` | Gets the feature importance scores. |
| `GetInputShape` | Saves the model to a file. |
| `GetModelMetadata` | Gets the model type. |
| `GetOutputShape` |  |
| `GetParameters` | Gets all model parameters as a single vector. |
| `GetUniqueEventTimes(Vector<>,Vector<Int32>)` | Gets unique sorted event times from the data. |
| `IsFeatureUsed(Int32)` | Determines whether a specific feature is used in the model. |
| `LoadModel(String)` | Loads the model from a file. |
| `LoadState(Stream)` | Loads the model's state from a stream. |
| `Predict(Matrix<>)` | Standard prediction - returns hazard ratios or survival at median time. |
| `PredictCumulativeHazard(Vector<>,Matrix<>)` | Predicts cumulative hazard at specified times (interface method). |
| `PredictHazardRatio(Matrix<>)` | Predicts hazard ratios relative to a baseline. |
| `PredictMedianSurvivalTime(Matrix<>)` | Gets the estimated median survival time (interface method). |
| `PredictRisk(Matrix<>)` | Predicts risk scores for subjects (interface method). |
| `PredictSurvival(Vector<>,Matrix<>)` | Predicts survival probability at specified times (interface method). |
| `PredictSurvivalProbability(Matrix<>,Vector<>)` | Predicts survival probabilities at specified time points. |
| `SanitizeParameters(Vector<>)` |  |
| `SaveState(Stream)` | Saves the model's state to a stream. |
| `Serialize` | Serializes the model to a byte array. |
| `SerializeInternalUnchecked` | Internal, non-virtual, no-guard serialization used by trusted framework call sites such as `DeepCopy`. |
| `SetActiveFeatureIndices(IEnumerable<Int32>)` | Sets the active feature indices for this model. |
| `SetParameters(Vector<>)` | Sets the parameters for this model. |
| `Train(Matrix<>,Vector<>)` | Standard model training - redirects to survival-specific training. |
| `ValidateSurvivalData(Matrix<>,Vector<>,Vector<Int32>)` | Validates survival data inputs. |
| `WithParameters(Vector<>)` | Creates a new instance of the model with specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `BaselineSurvivalFunction` | Stores the baseline survival function at each event time. |
| `IsFitted` | Indicates whether the model has been fitted. |
| `NumOps` | Numeric operations helper for generic math. |
| `TrainedEventTimes` | Stores the unique sorted event times from training data. |
| `_defaultLossFunction` | The default loss function for gradient computation. |

