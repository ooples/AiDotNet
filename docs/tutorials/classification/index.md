---
layout: default
title: Classification
parent: Tutorials
nav_order: 1
has_children: true
permalink: /tutorials/classification/
---

# Classification Tutorial
{: .no_toc }

Learn to build classification models with AiDotNet.
{: .fs-6 .fw-300 }

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## What is Classification?

Classification is a supervised learning task where the goal is to predict discrete class labels from input features. Examples include:
- Email spam detection (spam/not spam)
- Image recognition (cat/dog/bird)
- Disease diagnosis (positive/negative)

## Types of Classification

### Binary Classification
Exactly two classes. Example: Is this email spam?

### Multi-class Classification
Three or more classes. Example: What species is this flower?

### Multi-label Classification
Each sample can belong to multiple classes. Example: Tag this article with relevant topics.

---

## Quick Start

```csharp
using AiDotNet;
using AiDotNet.Classification;

// Prepare data
var features = new double[][]
{
    new[] { 5.1, 3.5, 1.4, 0.2 },  // Setosa
    new[] { 7.0, 3.2, 4.7, 1.4 },  // Versicolor
    new[] { 6.3, 3.3, 6.0, 2.5 }   // Virginica
};
var labels = new double[] { 0, 1, 2 };

// Build and train
var result = await new PredictionModelBuilder<double, double[], double>()
    .ConfigureModel(new RandomForestClassifier<double>(nEstimators: 100))
    .ConfigurePreprocessing()
    .ConfigureCrossValidation(new KFoldCrossValidator<double>(k: 5))
    .BuildAsync(features, labels);

// Evaluate
Console.WriteLine($"Accuracy: {result.CrossValidationResult?.MeanScore:P2}");

// Predict
var prediction = result.Model.Predict(new[] { 5.9, 3.0, 5.1, 1.8 });
Console.WriteLine($"Predicted class: {prediction}");
```

---

## Available Classifiers

### Tree-Based Methods

| Classifier | Description | Best For |
|:-----------|:------------|:---------|
| `RandomForestClassifier` | Ensemble of decision trees | General purpose, tabular data |
| `GradientBoostingClassifier` | Boosted decision trees | High accuracy, Kaggle competitions |
| `DecisionTreeClassifier` | Single decision tree | Interpretability |

### Linear Methods

| Classifier | Description | Best For |
|:-----------|:------------|:---------|
| `LogisticRegression` | Probabilistic linear classifier | Baseline, simple problems |
| `SVMClassifier` | Support Vector Machine | High-dimensional data |
| `RidgeClassifier` | L2-regularized linear | Multicollinearity |

### Bayesian Methods

| Classifier | Description | Best For |
|:-----------|:------------|:---------|
| `GaussianNaiveBayes` | Continuous features | Fast baseline |
| `MultinomialNaiveBayes` | Count features | Text classification |
| `BernoulliNaiveBayes` | Binary features | Document classification |

### Distance-Based

| Classifier | Description | Best For |
|:-----------|:------------|:---------|
| `KNearestNeighbors` | Instance-based learning | Small datasets, no training |

### Neural Networks

```csharp
var result = await new PredictionModelBuilder<float, Tensor<float>, Tensor<float>>()
    .ConfigureModel(new NeuralNetworkClassifier<float>(
        inputFeatures: 784,
        numClasses: 10,
        complexity: NetworkComplexity.Medium))
    .ConfigureOptimizer(new AdamOptimizer<float>(learningRate: 0.001f))
    .ConfigureGpuAcceleration()
    .BuildAsync(images, labels);
```

---

## Data Preprocessing

### Automatic Preprocessing

```csharp
.ConfigurePreprocessing()  // Applies StandardScaler by default
```

### Custom Preprocessing

```csharp
.ConfigurePreprocessing(new PreprocessingConfig
{
    Scaler = new MinMaxScaler<double>(),
    ImputeStrategy = ImputeStrategy.Mean,
    HandleCategorical = true
})
```

---

## Cross-Validation

Validate model performance with cross-validation:

```csharp
.ConfigureCrossValidation(new KFoldCrossValidator<double>(k: 5))

// Or stratified (maintains class distribution)
.ConfigureCrossValidation(new StratifiedKFoldCrossValidator<double>(k: 5))
```

Access results:
```csharp
var cv = result.CrossValidationResult;
Console.WriteLine($"Mean: {cv.MeanScore:P2}");
Console.WriteLine($"Std: {cv.StandardDeviation:P4}");
Console.WriteLine($"Folds: {string.Join(", ", cv.FoldScores.Select(s => $"{s:P2}"))}");
```

---

## Evaluation Metrics

### Confusion Matrix

```csharp
var predictions = testSamples.Select(s => result.Model.Predict(s)).ToArray();
var cm = ConfusionMatrix.Compute(predictions, testLabels);

Console.WriteLine($"Accuracy: {cm.Accuracy:P2}");
Console.WriteLine($"Precision: {cm.Precision:P2}");
Console.WriteLine($"Recall: {cm.Recall:P2}");
Console.WriteLine($"F1 Score: {cm.F1Score:P2}");
```

### ROC Curve (Binary)

```csharp
var rocCurve = ROCCurve.Compute(predictions, testLabels);
Console.WriteLine($"AUC: {rocCurve.AUC:F4}");
```

---

## Hyperparameter Tuning

### Grid Search

```csharp
var result = await new PredictionModelBuilder<double, double[], double>()
    .ConfigureModel(new RandomForestClassifier<double>())
    .ConfigureHyperparameterSearch(new GridSearchConfig
    {
        Parameters = new Dictionary<string, object[]>
        {
            ["nEstimators"] = new object[] { 50, 100, 200 },
            ["maxDepth"] = new object[] { 5, 10, null }
        },
        ScoringMetric = ScoringMetric.Accuracy
    })
    .BuildAsync(features, labels);
```

### AutoML

```csharp
.ConfigureAutoML(new AutoMLConfig<double>
{
    MaxTrials = 50,
    TimeoutMinutes = 30,
    Metric = AutoMLMetric.F1Score
})
```

---

## Best Practices

1. **Start Simple**: Use Logistic Regression as a baseline
2. **Check Class Balance**: Use stratified sampling for imbalanced data
3. **Feature Scaling**: Most algorithms benefit from scaled features
4. **Cross-Validate**: Always use CV, not a single train/test split
5. **Monitor Overfitting**: Compare train vs validation accuracy

---

## Common Issues

### Imbalanced Classes

```csharp
.ConfigureClassWeights(ClassWeightStrategy.Balanced)
// Or custom weights
.ConfigureClassWeights(new Dictionary<int, double> { [0] = 1.0, [1] = 10.0 })
```

### Overfitting

- Reduce model complexity
- Add regularization
- Increase training data
- Use dropout (for neural networks)

### Underfitting

- Increase model complexity
- Add more features
- Reduce regularization

---

## Next Steps

- [Binary Classification Sample](/samples/classification/BinaryClassification/SentimentAnalysis/)
- [Multi-class Classification Sample](/samples/classification/MultiClassification/IrisClassification/)
- [Classification API Reference](/api/AiDotNet.Classification/)
