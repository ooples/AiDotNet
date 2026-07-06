---
title: "Classical ML"
description: "Classical ML algorithms reference."
order: 2
section: "Reference"
---


Reference for the classical machine-learning algorithms in AiDotNet. They train through the same facade — `ConfigureModel(...)` + `ConfigureDataLoader(...)` + `BuildAsync()`. Supervised models (regression, classification) then predict labels/values through `result.Predict(...)`; clustering returns cluster assignments and dimensionality-reduction returns transformed features, so their output shape differs.

---

## Classification

### Linear Models

| Algorithm | Namespace |
|:----------|:----------|
| `LogisticRegression<T>` | `AiDotNet.Regression` |
| `RidgeClassifier<T>` | `AiDotNet.Classification` |
| `SGDClassifier<T>` | `AiDotNet.Classification` |
| `PassiveAggressiveClassifier<T>` | `AiDotNet.Classification` |
| `PerceptronClassifier<T>` | `AiDotNet.Classification` |

```csharp
using AiDotNet.Regression;

var classifier = new LogisticRegression<double>();
```

### Support Vector Machines

| Algorithm | Namespace |
|:----------|:----------|
| `SupportVectorClassifier<T>` | `AiDotNet.Classification.SVM` |
| `NuSupportVectorClassifier<T>` | `AiDotNet.Classification.SVM` |
| `LinearSupportVectorClassifier<T>` | `AiDotNet.Classification.SVM` |

```csharp
using AiDotNet.Classification.SVM;

var svm = new LinearSupportVectorClassifier<double>();
```

### Tree-Based & Ensemble

| Algorithm | Description |
|:----------|:------------|
| `DecisionTreeClassifier<T>` | Single decision tree |
| `RandomForestClassifier<T>` | Ensemble of trees |
| `ExtraTreesClassifier<T>` | Extremely randomized trees |
| `GradientBoostingClassifier<T>` | Gradient boosting |
| `HistGradientBoostingClassifier<T>` | Histogram-based boosting |
| `AdaBoostClassifier<T>` | Adaptive boosting |
| `BaggingClassifier<T>` | Bootstrap aggregating |
| `VotingClassifier<T>` | Soft/hard voting |
| `StackingClassifier<T>` | Stacked generalization |

```csharp
using AiDotNet.Classification.Ensemble;
using AiDotNet.Models.Options;

var forest = new RandomForestClassifier<double>(
    new RandomForestClassifierOptions<double> { NEstimators = 100, MaxDepth = 10 });
```

### Naive Bayes

| Algorithm | Distribution |
|:----------|:-------------|
| `GaussianNaiveBayes<T>` | Gaussian (continuous) |
| `MultinomialNaiveBayes<T>` | Multinomial (counts) |
| `BernoulliNaiveBayes<T>` | Bernoulli (binary) |
| `ComplementNaiveBayes<T>` | Complement (imbalanced) |
| `CategoricalNaiveBayes<T>` | Categorical |

### Neighbors & Multiclass

| Algorithm | Description |
|:----------|:------------|
| `KNeighborsClassifier<T>` | K-Nearest Neighbors |
| `OneVsRestClassifier<T>` | One-vs-rest wrapper |
| `OneVsOneClassifier<T>` | One-vs-one wrapper |
| `ClassifierChainClassifier<T>` | Multi-label chains |

---

## Regression

> AiDotNet regressors use the `Regression` suffix (e.g. `RidgeRegression`, not `Ridge`).

### Linear Models

| Algorithm | Description |
|:----------|:------------|
| `SimpleRegression<T>` / `MultipleRegression<T>` | Ordinary least squares |
| `RidgeRegression<T>` | L2 regularization |
| `LassoRegression<T>` | L1 regularization |
| `ElasticNetRegression<T>` | L1 + L2 regularization |
| `PolynomialRegression<T>` | Polynomial features |
| `BayesianRegression<T>` | Bayesian regression |
| `RobustRegression<T>` | Robust to outliers |
| `QuantileRegression<T>` | Quantile regression |

```csharp
using AiDotNet.Regression;

var model = new ElasticNetRegression<double>();
```

### Tree-Based, Neighbors & Neural

| Algorithm | Description |
|:----------|:------------|
| `DecisionTreeRegression<T>` | Single decision tree |
| `RandomForestRegression<T>` | Ensemble of trees |
| `GradientBoostingRegression<T>` | Gradient boosting |
| `HistGradientBoostingRegression<T>` | Histogram-based boosting |
| `KNearestNeighborsRegression<T>` | K-Nearest Neighbors |
| `GaussianProcessRegression<T>` | Gaussian process |
| `NeuralNetworkRegression<T>` | Multi-layer perceptron |
| `IsotonicRegression<T>` | Monotonic regression |

---

## Clustering

| Family | Algorithms |
|:-------|:-----------|
| Centroid-based | `KMeans<T>`, `MiniBatchKMeans<T>`, `KMedoids<T>`, `BisectingKMeans<T>`, `FuzzyCMeans<T>` |
| Density-based | `DBSCAN<T>`, `HDBSCAN<T>`, `OPTICS<T>`, `MeanShift<T>`, `Denclue<T>` |
| Hierarchical | `AgglomerativeClustering<T>`, `BIRCH<T>`, `CURE<T>` |
| Graph / other | `SpectralClustering<T>`, `AffinityPropagation<T>`, `CLARANS<T>` |

```csharp
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;

var kmeans = new KMeans<double>(new KMeansOptions<double> { NumClusters = 5 });
```

---

## Dimensionality Reduction

| Family | Algorithms |
|:-------|:-----------|
| Linear | `PCA<T>`, `IncrementalPCA<T>`, `KernelPCA<T>`, `FactorAnalysis<T>`, `FastICA<T>`, `LatentDirichletAllocation<T>` |
| Manifold | `Isomap<T>`, `LocallyLinearEmbedding<T>`, `MDS<T>`, `LaplacianEigenmaps<T>`, `DiffusionMaps<T>` |

---

## Usage with AiModelBuilder

Most algorithms above plug into the facade the same way — swap `ConfigureModel(...)`. Supervised models use a labelled loader (`FromArrays`/`FromMatrixVector`) and read predictions from `result.Predict(...)`; unsupervised ones (clustering, dimensionality reduction) use the features-only `DataLoaders.FromMatrix(...)` and return cluster labels / transformed features.

```csharp
using AiDotNet;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

double[][] features =
{
    new[] { 5.1, 3.5, 1.4, 0.2 }, new[] { 7.0, 3.2, 4.7, 1.4 },
    new[] { 6.3, 3.3, 6.0, 2.5 }, new[] { 4.9, 3.0, 1.4, 0.2 }
};
double[] labels = { 0, 1, 2, 0 };

// Classification
var classification = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new RandomForestClassifier<double>(
        new RandomForestClassifierOptions<double> { NEstimators = 100 }))
    .ConfigureDataLoader(DataLoaders.FromArrays(features, labels))
    .BuildAsync();

// Regression — same shape, different model + targets.
double[] targets = { 1.2, 3.4, 5.6, 1.1 };
var regression = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new GradientBoostingRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine($"Trained classifier + regressor on {features.Length} samples.");
```

```csharp
using AiDotNet;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.LinearAlgebra;

// Clustering uses the features-only loader.
var data = new Matrix<double>(4, 2);
double[][] rows = { new[] { 1.0, 1.0 }, new[] { 1.2, 0.9 }, new[] { 8.0, 8.0 }, new[] { 8.1, 7.9 } };
for (int i = 0; i < 4; i++) { data[i, 0] = rows[i][0]; data[i, 1] = rows[i][1]; }

var clustering = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new KMeans<double>(new KMeansOptions<double> { NumClusters = 2 }))
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

Console.WriteLine($"Silhouette: {clustering.Evaluation.ClusteringMetrics?.Silhouette:F4}");
```
