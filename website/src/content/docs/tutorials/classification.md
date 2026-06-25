---
title: "Classification"
description: "Build classification models with AiDotNet."
order: 1
section: "Tutorials"
---


Learn to build classification models with AiDotNet.

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

---

## Quick Start

```csharp
using AiDotNet;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

// Iris dataset samples: 4 features, 3 species (0=setosa, 1=versicolor, 2=virginica)
double[][] features =
{
    new[] { 5.1, 3.5, 1.4, 0.2 }, new[] { 4.9, 3.0, 1.4, 0.2 },
    new[] { 7.0, 3.2, 4.7, 1.4 }, new[] { 6.4, 3.2, 4.5, 1.5 },
    new[] { 6.3, 3.3, 6.0, 2.5 }, new[] { 5.8, 2.7, 5.1, 1.9 }
};
double[] labels = { 0, 0, 1, 1, 2, 2 };

var X = ToMatrix(features);
var y = new Vector<double>(labels);

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new RandomForestClassifier<double>(
        new RandomForestClassifierOptions<double> { NEstimators = 100 }))
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(X, y))
    .BuildAsync();

var newFlower = new Matrix<double>(1, 4);
foreach (var (v, j) in new[] { 5.9, 3.0, 5.1, 1.8 }.Select((v, j) => (v, j)))
    newFlower[0, j] = v;
Console.WriteLine($"Predicted class: {(int)result.Predict(newFlower)[0]}");

// NOTE: these are computed on the SAME data the model was fit on, so they are resubstitution
// (training-set) metrics — optimistic, not a validation estimate. Use a holdout split to judge
// generalization.
var stats = result.GetDataSetStats(X, y);
Console.WriteLine($"Training accuracy: {stats.ErrorStats.Accuracy:P2}, F1: {stats.ErrorStats.F1Score:P2}");

static Matrix<double> ToMatrix(double[][] rows)
{
    var m = new Matrix<double>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            m[i, j] = rows[i][j];
    return m;
}
```

---

## Available Classifiers

Swap the `ConfigureModel(...)` argument to change algorithm.

| Family | Classifiers |
|:-------|:------------|
| Tree-based | `RandomForestClassifier`, `GradientBoostingClassifier`, `DecisionTreeClassifier`, `ExtraTreesClassifier`, `AdaBoostClassifier` |
| Linear | `LogisticRegression`, `LinearSupportVectorClassifier` |
| Bayesian | `GaussianNaiveBayes`, `MultinomialNaiveBayes`, `BernoulliNaiveBayes`, `ComplementNaiveBayes` |
| Distance-based | `KNeighborsClassifier` |

### Naive Bayes

```csharp
using AiDotNet;
using AiDotNet.Classification.NaiveBayes;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.LinearAlgebra;

double[][] features =
{
    new[] { 5.1, 3.5, 1.4, 0.2 }, new[] { 4.9, 3.0, 1.4, 0.2 },
    new[] { 7.0, 3.2, 4.7, 1.4 }, new[] { 6.4, 3.2, 4.5, 1.5 },
    new[] { 6.3, 3.3, 6.0, 2.5 }, new[] { 5.8, 2.7, 5.1, 1.9 }
};
double[] labels = { 0, 0, 1, 1, 2, 2 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new GaussianNaiveBayes<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, labels))
    .BuildAsync();

Console.WriteLine("Trained a Gaussian Naive Bayes classifier.");
```

### Neural Network

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

var rng = new Random(42);
var trainX = new Tensor<double>(new[] { 120, 16 });
var trainY = new Tensor<double>(new[] { 120, 3 });
for (int i = 0; i < 120; i++)
{
    for (int j = 0; j < 16; j++) trainX[new[] { i, j }] = rng.NextDouble();
    trainY[new[] { i, i % 3 }] = 1.0;
}

var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
        inputFeatures: 16, numClasses: 3, complexity: NetworkComplexity.Medium)))
    .ConfigureGpuAcceleration()
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine($"Output shape: [{string.Join(", ", result.Predict(trainX).Shape)}]");
```

---

## Cross-Validation

Validate with k-fold cross-validation (a `StratifiedKFoldCrossValidator` is also available for preserving class balance):

```csharp
using AiDotNet;
using AiDotNet.Classification.Ensemble;
using AiDotNet.CrossValidators;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

double[][] features =
{
    new[] { 5.1, 3.5, 1.4, 0.2 }, new[] { 4.9, 3.0, 1.4, 0.2 }, new[] { 5.0, 3.4, 1.5, 0.2 },
    new[] { 7.0, 3.2, 4.7, 1.4 }, new[] { 6.4, 3.2, 4.5, 1.5 }, new[] { 6.9, 3.1, 4.9, 1.5 },
    new[] { 6.3, 3.3, 6.0, 2.5 }, new[] { 5.8, 2.7, 5.1, 1.9 }, new[] { 7.1, 3.0, 5.9, 2.1 }
};
double[] labels = { 0, 0, 0, 1, 1, 1, 2, 2, 2 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new RandomForestClassifier<double>(
        new RandomForestClassifierOptions<double> { NEstimators = 50 }))
    .ConfigureCrossValidation(new KFoldCrossValidator<double, Matrix<double>, Vector<double>>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, labels))
    .BuildAsync();

if (result.CrossValidationResult is not null)
    Console.WriteLine($"Cross-validation folds: {result.CrossValidationResult.FoldResults.Count}");
```

---

## Evaluation Metrics

`result.GetDataSetStats(X, y).ErrorStats` carries every classification metric — accuracy, precision, recall, F1, and AUC.

```csharp
using AiDotNet;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

double[][] data =
{
    new[] { 0.1, 0.9 }, new[] { 0.2, 0.8 }, new[] { 0.15, 0.85 },
    new[] { 0.9, 0.1 }, new[] { 0.8, 0.2 }, new[] { 0.85, 0.15 }
};
double[] labels = { 0, 0, 0, 1, 1, 1 };

var X = ToMatrix(data);
var y = new Vector<double>(labels);

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new RandomForestClassifier<double>(
        new RandomForestClassifierOptions<double> { NEstimators = 50 }))
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(X, y))
    .BuildAsync();

var s = result.GetDataSetStats(X, y).ErrorStats;
Console.WriteLine($"Accuracy:  {s.Accuracy:P2}");
Console.WriteLine($"Precision: {s.Precision:P2}");
Console.WriteLine($"Recall:    {s.Recall:P2}");
Console.WriteLine($"F1 Score:  {s.F1Score:P2}");
Console.WriteLine($"AUC-ROC:   {s.AUCROC:F4}");

static Matrix<double> ToMatrix(double[][] rows)
{
    var m = new Matrix<double>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            m[i, j] = rows[i][j];
    return m;
}
```

---

## Best Practices

1. **Start simple**: `LogisticRegression` or `GaussianNaiveBayes` make strong baselines.
2. **Check class balance**: use `StratifiedKFoldCrossValidator` for imbalanced data.
3. **Feature scaling**: linear and distance-based models benefit from `ConfigurePreprocessing(...)`.
4. **Cross-validate**: prefer cross-validation over a single train/test split.
5. **Watch overfitting**: compare `result.Evaluation.TrainingSet` against `result.Evaluation.ValidationSet`.

---

## Next Steps

- [Regression Tutorial](/docs/tutorials/regression/) — for predicting continuous values
- [NLP Tutorial](/docs/tutorials/nlp/) — for text classification
