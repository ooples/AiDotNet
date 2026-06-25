---
title: "CausalModelBase<T>"
description: "Abstract base class for causal inference models."
section: "API Reference"
---

`Base Classes` · `AiDotNet.CausalInference`

Abstract base class for causal inference models.

## For Beginners

This class contains shared code that all causal inference models need,
so each specific model (like PropensityScoreMatching) doesn't have to reimplement it.

Key shared functionality:

- Estimating propensity scores (probability of treatment)
- Calculating treatment effects
- Checking overlap assumptions
- Managing fitted model state

## How It Works

This base class provides common functionality for causal inference models including
propensity score estimation, treatment effect calculation, and overlap checking.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CausalModelBase` | Initializes a new instance of the CausalModelBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` | Gets the default loss function. |
| `Engine` | Gets the hardware-accelerated computation engine for vectorized operations. |
| `FeatureNames` | Gets or sets the feature names. |
| `IsTrained` | Gets whether the model is trained. |
| `NumFeatures` | Gets the number of features the model was trained on. |
| `ParameterCount` | Gets the total number of parameters in the model. |
| `SupportsParameterInitialization` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies gradients to update model parameters. |
| `CalculateBootstrapStandardError(Func<Matrix<>,Vector<Int32>,Vector<>,>,Matrix<>,Vector<Int32>,Vector<>,Int32)` | Calculates the standard error using bootstrap resampling. |
| `CheckOverlap(Matrix<>,Vector<Int32>)` | Checks the overlap/positivity assumption. |
| `Clone` | Creates a clone of the model. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` | Computes gradients for the given input and target. |
| `CreateNewInstance` | Creates a new instance of the same type. |
| `DeepCopy` | Creates a deep copy of the model. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `DeserializeInternalUnchecked(Byte[])` | Internal, non-virtual, no-guard deserialization used by trusted framework call sites such as `DeepCopy`. |
| `Dispose` |  |
| `Dispose(Boolean)` | Releases resources held by this causal model. |
| `EnsureFitted` | Ensures the model has been fitted before making predictions. |
| `EstimateATE(Matrix<>,Vector<Int32>,Vector<>)` | Estimates the Average Treatment Effect (ATE) from the data with standard error. |
| `EstimateATT(Matrix<>,Vector<Int32>,Vector<>)` | Estimates the Average Treatment Effect on the Treated (ATT). |
| `EstimateAverageTreatmentEffect(Matrix<>)` | Estimates the Average Treatment Effect (ATE) across the population. |
| `EstimateCATEPerIndividual(Matrix<>,Vector<Int32>,Vector<>)` | Estimates the Conditional Average Treatment Effect (CATE) for each individual. |
| `EstimatePropensityScores(Matrix<>)` | Estimates propensity scores for each individual. |
| `EstimatePropensityScoresCore(Matrix<>)` | Core propensity score estimation to be implemented by derived classes. |
| `EstimateTreatmentEffect(Matrix<>)` | Estimates the Conditional Average Treatment Effect (CATE) for subjects. |
| `Fit(Matrix<>,Vector<>,Vector<>)` | Fits the causal model to observational data. |
| `FitLogisticRegression(Matrix<>,Vector<Int32>,Int32,Double)` | Fits a simple logistic regression for propensity score estimation. |
| `GetActiveFeatureIndices` | Gets the indices of features that are actively used in the model. |
| `GetAdditionalModelData` | Gets additional model data to include in serialization. |
| `GetDynamicShapeInfo` |  |
| `GetFeatureImportance` | Gets the feature importance scores. |
| `GetInputShape` | Saves the model to a file. |
| `GetModelMetadata` | Gets the model type. |
| `GetOutputShape` |  |
| `GetParameters` | Gets all model parameters as a single vector. |
| `IsFeatureUsed(Int32)` | Determines whether a specific feature is used in the model. |
| `LoadAdditionalModelData(JObject)` | Loads additional model data from deserialization. |
| `LoadModel(String)` | Loads the model from a file. |
| `LoadState(Stream)` | Loads the model's state from a stream. |
| `Predict(Matrix<>)` | Standard prediction - returns predicted outcomes. |
| `PredictControl(Matrix<>)` | Predicts the outcome under control. |
| `PredictPropensityWithCoefficients(Matrix<>,Vector<>)` | Predicts propensity scores using fitted logistic regression coefficients. |
| `PredictTreated(Matrix<>)` | Predicts the outcome under treatment. |
| `PredictTreatmentEffect(Matrix<>)` | Predicts the treatment effect for new individuals. |
| `SanitizeParameters(Vector<>)` |  |
| `SaveState(Stream)` | Saves the model's state to a stream. |
| `Serialize` | Serializes the model to a byte array. |
| `SerializeInternalUnchecked` | Internal, non-virtual, no-guard serialization used by trusted framework call sites such as `DeepCopy`. |
| `SetActiveFeatureIndices(IEnumerable<Int32>)` | Sets the active feature indices for this model. |
| `SetParameters(Vector<>)` | Sets the parameters for this model. |
| `Sigmoid(Double)` | Sigmoid function for logistic regression. |
| `ThrowIfDisposed` | Throws `ObjectDisposedException` if `Dispose` has already been called. |
| `Train(Matrix<>,Vector<>)` | Standard model training - fits the causal model. |
| `ValidateCausalData(Matrix<>,Vector<Int32>,Vector<>)` | Validates causal inference data inputs. |
| `WithParameters(Vector<>)` | Creates a new instance of the model with specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `IsFitted` | Indicates whether the model has been fitted. |
| `NumOps` | Numeric operations helper for generic math. |
| `_defaultLossFunction` | The default loss function for gradient computation. |

