---
layout: default
title: Classical ML
parent: Reference
nav_order: 2
permalink: /reference/classical-ml/
---

# Classical Machine Learning
{: .no_toc }

Complete reference for all 106+ classical ML algorithms in AiDotNet.
{: .fs-6 .fw-300 }

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Classification (28 Algorithms)

### Linear Models

| Algorithm | Description | Multi-class |
|:----------|:------------|:------------|
| `LogisticRegression<T>` | Binary/multi-class logistic | Yes |
| `LinearSVC<T>` | Linear Support Vector Classifier | Yes |
| `RidgeClassifier<T>` | Ridge regression for classification | Yes |
| `SGDClassifier<T>` | Stochastic Gradient Descent | Yes |
| `Perceptron<T>` | Linear perceptron | Yes |

```csharp
var classifier = new LogisticRegression<double>(
    regularization: 0.01,
    solver: LogisticSolver.LBFGS,
    maxIterations: 1000);
```

### Support Vector Machines

| Algorithm | Description | Kernel |
|:----------|:------------|:-------|
| `SVC<T>` | Support Vector Classifier | Multiple |
| `NuSVC<T>` | Nu-Support Vector Classifier | Multiple |
| `LinearSVC<T>` | Linear SVC | Linear |

Supported kernels: `Linear`, `Polynomial`, `RBF`, `Sigmoid`, `Precomputed`

```csharp
var svm = new SVC<double>(
    kernel: KernelType.RBF,
    C: 1.0,
    gamma: "scale");
```

### Tree-Based

| Algorithm | Description |
|:----------|:------------|
| `DecisionTreeClassifier<T>` | Single decision tree |
| `RandomForestClassifier<T>` | Ensemble of trees |
| `ExtraTreesClassifier<T>` | Extremely randomized trees |
| `GradientBoostingClassifier<T>` | Gradient boosting |
| `XGBoostClassifier<T>` | XGBoost implementation |
| `LightGBMClassifier<T>` | LightGBM implementation |
| `CatBoostClassifier<T>` | CatBoost implementation |

```csharp
var forest = new RandomForestClassifier<double>(
    nEstimators: 100,
    maxDepth: 10,
    minSamplesSplit: 2);
```

### Naive Bayes

| Algorithm | Distribution |
|:----------|:-------------|
| `GaussianNB<T>` | Gaussian (continuous) |
| `MultinomialNB<T>` | Multinomial (counts) |
| `BernoulliNB<T>` | Bernoulli (binary) |
| `ComplementNB<T>` | Complement (imbalanced) |

### Neighbors

| Algorithm | Description |
|:----------|:------------|
| `KNeighborsClassifier<T>` | K-Nearest Neighbors |
| `RadiusNeighborsClassifier<T>` | Radius-based neighbors |

### Neural Networks

| Algorithm | Description |
|:----------|:------------|
| `MLPClassifier<T>` | Multi-layer Perceptron |

### Ensemble

| Algorithm | Description |
|:----------|:------------|
| `BaggingClassifier<T>` | Bootstrap aggregating |
| `AdaBoostClassifier<T>` | Adaptive Boosting |
| `VotingClassifier<T>` | Soft/hard voting |
| `StackingClassifier<T>` | Stacked generalization |

---

## Regression (41 Algorithms)

### Linear Models

| Algorithm | Description |
|:----------|:------------|
| `LinearRegression<T>` | Ordinary Least Squares |
| `Ridge<T>` | L2 regularization |
| `Lasso<T>` | L1 regularization |
| `ElasticNet<T>` | L1 + L2 regularization |
| `Lars<T>` | Least Angle Regression |
| `LassoLars<T>` | LARS with L1 |
| `OrthogonalMatchingPursuit<T>` | Sparse approximation |
| `BayesianRidge<T>` | Bayesian regression |
| `ARDRegression<T>` | Automatic Relevance Determination |
| `SGDRegressor<T>` | SGD for regression |
| `HuberRegressor<T>` | Robust to outliers |
| `RANSACRegressor<T>` | RANSAC robust regression |
| `TheilSenRegressor<T>` | Theil-Sen estimator |
| `PassiveAggressiveRegressor<T>` | Online learning |

```csharp
var model = new ElasticNet<double>(
    alpha: 1.0,
    l1Ratio: 0.5,
    maxIterations: 1000);
```

### Support Vector Regression

| Algorithm | Description |
|:----------|:------------|
| `SVR<T>` | Support Vector Regression |
| `NuSVR<T>` | Nu-Support Vector Regression |
| `LinearSVR<T>` | Linear SVR |

### Tree-Based

| Algorithm | Description |
|:----------|:------------|
| `DecisionTreeRegressor<T>` | Single decision tree |
| `RandomForestRegressor<T>` | Ensemble of trees |
| `ExtraTreesRegressor<T>` | Extremely randomized trees |
| `GradientBoostingRegressor<T>` | Gradient boosting |
| `XGBoostRegressor<T>` | XGBoost implementation |
| `LightGBMRegressor<T>` | LightGBM implementation |
| `CatBoostRegressor<T>` | CatBoost implementation |

### Neighbors

| Algorithm | Description |
|:----------|:------------|
| `KNeighborsRegressor<T>` | K-Nearest Neighbors |
| `RadiusNeighborsRegressor<T>` | Radius-based neighbors |

### Gaussian Processes

| Algorithm | Description |
|:----------|:------------|
| `GaussianProcessRegressor<T>` | GP regression |

### Neural Networks

| Algorithm | Description |
|:----------|:------------|
| `MLPRegressor<T>` | Multi-layer Perceptron |

### Ensemble

| Algorithm | Description |
|:----------|:------------|
| `BaggingRegressor<T>` | Bootstrap aggregating |
| `AdaBoostRegressor<T>` | Adaptive Boosting |
| `VotingRegressor<T>` | Average predictions |
| `StackingRegressor<T>` | Stacked generalization |

### Isotonic

| Algorithm | Description |
|:----------|:------------|
| `IsotonicRegression<T>` | Monotonic regression |

### Quantile

| Algorithm | Description |
|:----------|:------------|
| `QuantileRegressor<T>` | Quantile regression |

---

## Clustering (20+ Algorithms)

### Centroid-Based

| Algorithm | Description |
|:----------|:------------|
| `KMeans<T>` | K-Means clustering |
| `MiniBatchKMeans<T>` | Mini-batch K-Means |
| `KMedoids<T>` | K-Medoids (PAM) |
| `BisectingKMeans<T>` | Bisecting K-Means |

```csharp
var kmeans = new KMeans<double>(
    nClusters: 5,
    maxIterations: 300,
    init: KMeansInit.KMeansPlusPlus);
```

### Density-Based

| Algorithm | Description |
|:----------|:------------|
| `DBSCAN<T>` | Density-based spatial clustering |
| `HDBSCAN<T>` | Hierarchical DBSCAN |
| `OPTICS<T>` | Ordering Points for clustering |
| `MeanShift<T>` | Mean shift clustering |

### Hierarchical

| Algorithm | Description |
|:----------|:------------|
| `AgglomerativeClustering<T>` | Bottom-up hierarchical |
| `BIRCH<T>` | Balanced Iterative Reducing |
| `Ward<T>` | Ward's minimum variance |

### Spectral

| Algorithm | Description |
|:----------|:------------|
| `SpectralClustering<T>` | Graph-based clustering |
| `SpectralBiclustering<T>` | Biclustering |
| `SpectralCoclustering<T>` | Co-clustering |

### Model-Based

| Algorithm | Description |
|:----------|:------------|
| `GaussianMixture<T>` | GMM clustering |
| `BayesianGaussianMixture<T>` | Bayesian GMM |

### Other

| Algorithm | Description |
|:----------|:------------|
| `AffinityPropagation<T>` | Message passing |
| `CLARA<T>` | Clustering Large Applications |
| `CLARANS<T>` | Randomized CLARA |

---

## Dimensionality Reduction

### Linear Methods

| Algorithm | Description |
|:----------|:------------|
| `PCA<T>` | Principal Component Analysis |
| `IncrementalPCA<T>` | Online PCA |
| `KernelPCA<T>` | Kernel PCA |
| `SparsePCA<T>` | Sparse PCA |
| `TruncatedSVD<T>` | Truncated SVD |
| `FactorAnalysis<T>` | Factor Analysis |
| `FastICA<T>` | Independent Component Analysis |
| `NMF<T>` | Non-negative Matrix Factorization |
| `LatentDirichletAllocation<T>` | Topic modeling |

### Manifold Learning

| Algorithm | Description |
|:----------|:------------|
| `TSNE<T>` | t-SNE visualization |
| `UMAP<T>` | Uniform Manifold Approximation |
| `Isomap<T>` | Isometric Mapping |
| `LocallyLinearEmbedding<T>` | LLE |
| `MDS<T>` | Multidimensional Scaling |
| `SpectralEmbedding<T>` | Spectral embedding |

---

## Anomaly Detection

| Algorithm | Description |
|:----------|:------------|
| `IsolationForest<T>` | Isolation-based |
| `LocalOutlierFactor<T>` | Density-based |
| `OneClassSVM<T>` | One-class SVM |
| `EllipticEnvelope<T>` | Gaussian distribution |
| `ECOD<T>` | Empirical Cumulative Distribution |

---

## Usage with PredictionModelBuilder

```csharp
// Classification
var result = await new PredictionModelBuilder<double, double[], int>()
    .ConfigureModel(new RandomForestClassifier<double>(nEstimators: 100))
    .ConfigurePreprocessing()
    .BuildAsync(features, labels);

// Regression
var result = await new PredictionModelBuilder<double, double[], double>()
    .ConfigureModel(new GradientBoostingRegressor<double>())
    .ConfigurePreprocessing()
    .BuildAsync(features, targets);

// Clustering
var result = await new PredictionModelBuilder<double, double[], int>()
    .ConfigureModel(new HDBSCAN<double>(minClusterSize: 5))
    .BuildAsync(features);
```
