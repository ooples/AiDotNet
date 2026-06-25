---
title: "ClusteringBase<T>"
description: "Provides a base implementation for clustering algorithms that group similar data points together."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Clustering.Base`

Provides a base implementation for clustering algorithms that group similar data points together.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ClusteringBase(ClusteringOptions<>,IRegularization<,Matrix<>,Vector<>>,ILossFunction<>)` | Initializes a new instance of the ClusteringBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClusterCenters` |  |
| `DefaultLossFunction` |  |
| `Engine` | Gets the global execution engine for vector operations. |
| `ExpectedParameterCount` | Gets the expected parameter count. |
| `FeatureNames` | Gets or sets the feature names. |
| `Inertia` |  |
| `IsTrained` | Gets or sets whether the model has been trained. |
| `Labels` |  |
| `NumClusters` |  |
| `NumFeatures` | Gets or sets the number of features. |
| `NumOps` | Gets the numeric operations for the specified type T. |
| `Options` | Gets the clustering options. |
| `ParameterCount` |  |
| `Random` | Random number generator. |
| `Regularization` | Gets the regularization method. |
| `SupportsParameterInitialization` | Whether this clustering model supports direct parameter-based initialization. |
| `TrainingDataRef` | Reference to the training data matrix. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `Clone` |  |
| `ComputeDistance(Matrix<>,Int32,Matrix<>,Int32)` | Computes distance between a sample and a cluster center. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` |  |
| `ComputeInertia(Matrix<>,Vector<>,Matrix<>)` | Computes inertia. |
| `ComputeSquaredDistance(Matrix<>,Int32,Matrix<>,Int32)` | Computes squared distance. |
| `CreateNewInstance` | Creates a new instance of this clustering algorithm. |
| `DeepCopy` |  |
| `Deserialize(Byte[])` |  |
| `Dispose` |  |
| `Dispose(Boolean)` | Releases resources held by this clustering model. |
| `Fit(Matrix<>)` | Fits the model on data (unsupervised). |
| `FitPredict(Matrix<>)` |  |
| `GetActiveFeatureIndices` |  |
| `GetDynamicShapeInfo` |  |
| `GetFeatureImportance` |  |
| `GetFeatureImportances` |  |
| `GetInputShape` |  |
| `GetModelMetadata` | Returns the model type identifier. |
| `GetOptions` |  |
| `GetOutputShape` |  |
| `GetParameters` |  |
| `GetRow(Matrix<>,Int32)` | Gets a row from a matrix. |
| `IsFeatureUsed(Int32)` |  |
| `LoadCheckpoint(String)` |  |
| `LoadModel(String)` |  |
| `LoadState(Stream)` |  |
| `MergeDegenerateClusters(Matrix<>)` | Merges clusters whose centers are within a small distance of each other. |
| `Predict(Matrix<>)` |  |
| `SanitizeParameters(Vector<>)` |  |
| `SaveCheckpoint(String)` |  |
| `SaveModel(String)` |  |
| `SaveState(Stream)` |  |
| `Serialize` |  |
| `SetActiveFeatureIndices(IEnumerable<Int32>)` |  |
| `SetParameters(Vector<>)` |  |
| `SetRow(Matrix<>,Int32,Vector<>)` | Sets a row in a matrix. |
| `Train(Matrix<>)` | Trains the model on data (unsupervised convenience method). |
| `Train(Matrix<>,Vector<>)` |  |
| `Transform(Matrix<>)` |  |
| `ValidateIsTrained` | Validates that the model has been trained. |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_defaultLossFunction` | Gets the default loss function. |

