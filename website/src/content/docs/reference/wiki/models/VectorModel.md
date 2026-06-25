---
title: "VectorModel<T>"
description: "Represents a linear model that uses a vector of coefficients to make predictions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Represents a linear model that uses a vector of coefficients to make predictions.

## For Beginners

This is a simple linear model that makes predictions by multiplying each input by a weight and adding them up.

When using this model:

- Each input feature has a corresponding coefficient (weight)
- Predictions are made by multiplying each input by its coefficient and summing the results
- The model can be trained using linear regression on example data
- It supports genetic algorithm operations for optimization

For example, if predicting house prices, the model might learn that:
price = 50,000 * bedrooms + 100 * square_feet + 20,000 * bathrooms

This is one of the simplest and most interpretable machine learning models,
making it a good starting point for many problems.

## How It Works

This class implements a simple linear model where predictions are made by computing the dot product of the input 
features and a vector of coefficients. It provides methods for training the model using linear regression, 
evaluating predictions, and genetic algorithm operations like mutation and crossover. This model is useful for 
linear regression problems and can serve as a building block for more complex models.

## Properties

| Property | Summary |
|:-----|:--------|
| `Coefficients` | Gets the vector of coefficients used by the model. |
| `Complexity` | Gets the complexity of the model. |
| `DefaultLossFunction` | Gets the default loss function used by this model for gradient computation. |
| `FeatureCount` | Gets the number of features used by the model. |
| `ParameterCount` | Gets the number of trainable parameters in the model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies pre-computed gradients to update the model parameters (coefficients). |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` | Computes gradients of the loss function with respect to model parameters WITHOUT updating parameters. |
| `ComputePDCombinations(Vector<Int32>,List<Vector<>>,Int32,[],Vector<>,Int32)` | Helper method to recursively compute partial dependence combinations. |
| `ConfigureFairness(Vector<Int32>,FairnessMetric[])` | Configures fairness evaluation settings. |
| `DeepCopy` | Creates a deep copy of this model. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `EnableMethod(InterpretationMethod[])` | Enables specific interpretation methods. |
| `Evaluate(Vector<>)` | Evaluates the model for a given input vector. |
| `GenerateTextExplanationAsync(Matrix<>,Vector<>)` | Generates a text explanation for a prediction. |
| `GenerateTextExplanationAsync(Tensor<>,Tensor<>)` | Generates a text explanation for a prediction (IInterpretableModel implementation). |
| `GetActiveFeatureIndices` | Gets the indices of all features used by this model. |
| `GetAnchorExplanationAsync(Matrix<>,)` | Gets anchor explanation for a given input. |
| `GetAnchorExplanationAsync(Tensor<>,)` | Gets anchor explanation for a given input (IInterpretableModel implementation). |
| `GetCounterfactualAsync(Matrix<>,Vector<>,Int32)` | Gets counterfactual explanation for a given input and desired output. |
| `GetCounterfactualAsync(Tensor<>,Tensor<>,Int32)` | Gets counterfactual explanation for a given input and desired output (IInterpretableModel implementation). |
| `GetDeepLIFTAsync(Tensor<>,Tensor<>,Boolean)` | Gets DeepLIFT attributions for a prediction. |
| `GetFeatureImportance` | Gets the feature importance scores. |
| `GetFeatureInteractionAsync(Int32,Int32)` | Gets feature interaction effects between two features. |
| `GetGlobalFeatureImportanceAsync` | Gets the global feature importance across all predictions. |
| `GetGradCAMAsync(Tensor<>,Int32)` | Gets GradCAM visual explanation for a prediction. |
| `GetIntegratedGradientsAsync(Tensor<>,Tensor<>,Int32)` | Gets Integrated Gradients attributions for a prediction. |
| `GetLimeExplanationAsync(Matrix<>,Int32)` | Gets LIME explanation for a specific input. |
| `GetLimeExplanationAsync(Tensor<>,Int32)` | Gets LIME explanation for a specific input (IInterpretableModel implementation). |
| `GetLocalFeatureImportanceAsync(Matrix<>)` | Gets the local feature importance for a specific input. |
| `GetLocalFeatureImportanceAsync(Tensor<>)` | Gets the local feature importance for a specific input (IInterpretableModel implementation). |
| `GetModelMetadata` | Gets metadata about the model. |
| `GetModelSpecificInterpretabilityAsync` | Gets model-specific interpretability information. |
| `GetParameters` | Gets all trainable parameters of the model as a single vector. |
| `GetPartialDependenceAsync(Vector<Int32>,Int32)` | Gets partial dependence data for specified features. |
| `GetShapValuesAsync(Matrix<>)` | Gets SHAP values for the given inputs. |
| `GetShapValuesAsync(Tensor<>)` | Gets SHAP values for the given inputs (IInterpretableModel implementation). |
| `IsFeatureUsed(Int32)` | Determines whether a specific feature is used by the model. |
| `LoadModel(String)` | Loads a model from a file into the current instance. |
| `LoadState(Stream)` | Loads the model's state (parameters and configuration) from a stream. |
| `Predict(Matrix<>)` | Predicts outputs for the provided input. |
| `PredictInternal(Matrix<>)` | Predicts the outputs for multiple input rows in a matrix. |
| `SaveModel(String)` | Saves the model to a file. |
| `SaveState(Stream)` | Saves the model's current state (parameters and configuration) to a stream. |
| `Serialize` | Serializes the model to a byte array. |
| `SetBaseModel(IFullModel<,,>)` | Sets the base model for interpretability analysis (IInterpretableModel implementation). |
| `SetBaseModel(IFullModel<,Matrix<>,Vector<>>)` | Sets the base model for interpretability analysis. |
| `SetParameters(Vector<>)` | Sets the parameters of the model. |
| `Train(Matrix<>,Vector<>)` | Trains the model on the provided generic input and expected output. |
| `TrainInternal(Matrix<>,Vector<>)` | Trains the model on the provided data using linear regression. |
| `ValidateFairnessAsync(Matrix<>,Int32)` | Validates fairness metrics for the given inputs. |
| `ValidateFairnessAsync(Tensor<>,Int32)` | Validates fairness metrics for the given inputs (IInterpretableModel implementation). |
| `WithParameters(Vector<>)` | Updates the model with new parameter values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_cachedFeatureImportance` | Cached feature importance to avoid recreating on every GetModelMetadata() call. |
| `_defaultLossFunction` | The default loss function used by this model for gradient computation. |

