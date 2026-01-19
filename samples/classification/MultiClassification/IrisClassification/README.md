# Iris Classification - Multi-Class Classification

This sample demonstrates multi-class classification using the classic Iris flower dataset, comparing multiple classifiers with 5-fold cross-validation.

## What You'll Learn

- How to compare multiple classifiers on the same dataset
- How to implement k-fold cross-validation
- Understanding multi-class classification metrics
- Interpreting per-class precision, recall, and F1-scores
- Reading multi-class confusion matrices

## The Iris Dataset

The Iris dataset is one of the most famous datasets in machine learning:

| Feature | Description | Range |
|---------|-------------|-------|
| sepal_length | Sepal length in cm | 4.3 - 7.9 |
| sepal_width | Sepal width in cm | 2.0 - 4.4 |
| petal_length | Petal length in cm | 1.0 - 6.9 |
| petal_width | Petal width in cm | 0.1 - 2.5 |

### The Three Classes

| Class | Description |
|-------|-------------|
| **Setosa** (0) | Small petals, easily separable |
| **Versicolor** (1) | Medium-sized, overlaps with Virginica |
| **Virginica** (2) | Largest flowers, overlaps with Versicolor |

## Classifiers Compared

### 1. Random Forest

```csharp
new RandomForestClassifier<double>(new RandomForestClassifierOptions<double>
{
    NEstimators = 100,
    MaxDepth = 10,
    MinSamplesSplit = 2,
    MaxFeatures = "sqrt",
    RandomState = 42
});
```

**Strengths**: Robust, handles non-linear boundaries, provides feature importance

### 2. Gradient Boosting

```csharp
new GradientBoostingClassifier<double>(new GradientBoostingClassifierOptions<double>
{
    NEstimators = 100,
    LearningRate = 0.1,
    MaxDepth = 3,
    Subsample = 0.8,
    RandomState = 42
});
```

**Strengths**: Often achieves best accuracy, sequential error correction

### 3. Support Vector Machine (RBF Kernel)

```csharp
new SupportVectorClassifier<double>(new SVMOptions<double>
{
    C = 1.0,
    Kernel = KernelType.RBF,
    Gamma = 0.1,
    RandomState = 42
});
```

**Strengths**: Effective in high-dimensional spaces, memory efficient

## Running the Sample

```bash
dotnet run
```

## Expected Output

```
=== AiDotNet Iris Classification ===
Multi-class classification comparing multiple classifiers with 5-fold cross-validation

Dataset: Iris flower classification
  - 150 samples, 4 features, 3 classes
  - Features: sepal_length, sepal_width, petal_length, petal_width
  - Classes: Setosa, Versicolor, Virginica

Class distribution:
  - Setosa: 50 samples
  - Versicolor: 50 samples
  - Virginica: 50 samples

Training set: 120 samples
Test set: 30 samples

======================================================
              5-FOLD CROSS-VALIDATION COMPARISON
======================================================

Training Random Forest...
  CV Accuracy: 95.83% (+/- 3.33%)
  Test Accuracy: 96.67%

Training Gradient Boosting...
  CV Accuracy: 96.67% (+/- 2.72%)
  Test Accuracy: 96.67%

Training SVM (RBF)...
  CV Accuracy: 95.00% (+/- 4.08%)
  Test Accuracy: 93.33%

======================================================
                  RESULTS SUMMARY
======================================================

Cross-Validation Results (5-Fold):
----------------------------------------------------------------------
Classifier           Fold 1     Fold 2     Fold 3     Fold 4     Fold 5
----------------------------------------------------------------------
Random Forest        96%        92%        96%        100%       96%
Gradient Boosting    96%        96%        96%        100%       96%
SVM (RBF)            92%        92%        96%        100%       96%

Summary Statistics:
-------------------------------------------------------
Classifier           Mean CV Acc      Std Dev      Test Acc
-------------------------------------------------------
Gradient Boosting    96.67%           2.72%        96.67%
Random Forest        95.83%           3.33%        96.67%
SVM (RBF)            95.00%           4.08%        93.33%

Best performing model: Gradient Boosting (Test Accuracy: 96.67%)

======================================================
          DETAILED CONFUSION MATRIX (GRADIENT BOOSTING)
======================================================

Confusion Matrix:
--------------------------------------------------
Actual \ Predicted    Setosa     Versicolor   Virginica
--------------------------------------------------
Setosa                10         0            0
Versicolor            0          9            1
Virginica             0          0            10

Per-Class Metrics:
------------------------------------------------------------
Class        Precision    Recall       F1-Score     Support
------------------------------------------------------------
Setosa       100.00%      100.00%      100.00%      10
Versicolor   100.00%      90.00%       94.74%       10
Virginica    90.91%       100.00%      95.24%       10
------------------------------------------------------------
Macro Avg    -            -            96.66%       30

Sample Predictions:
------------------------------------------------------------
  Sample  1: Predicted=Versicolor   Actual=Versicolor   [correct]
  Sample  2: Predicted=Setosa       Actual=Setosa       [correct]
  Sample  3: Predicted=Virginica    Actual=Virginica    [correct]
  ...

=== Sample Complete ===
```

## Understanding the Metrics

### Cross-Validation Scores

Each fold shows how well the model generalizes:
- **High variance** (different fold scores): Model may be overfitting
- **Low variance** (consistent fold scores): Model is stable

### Per-Class Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Precision** | TP / (TP + FP) | Of predicted class X, how many are correct |
| **Recall** | TP / (TP + FN) | Of actual class X, how many were found |
| **F1-Score** | 2 * P * R / (P + R) | Harmonic mean of precision and recall |
| **Support** | TP + FN | Number of samples in this class |

### Multi-Class Confusion Matrix

```
                  Predicted
              A     B     C
Actual A     [TP_A] [e]   [e]
Actual B     [e]   [TP_B] [e]
Actual C     [e]   [e]   [TP_C]
```

- **Diagonal**: Correct predictions (True Positives for each class)
- **Off-diagonal**: Misclassifications (errors)

## Code Highlights

### Manual 5-Fold Cross-Validation

```csharp
for (int fold = 0; fold < 5; fold++)
{
    var foldValidationIndices = Enumerable.Range(fold * foldSize, foldSize).ToArray();
    var foldTrainIndices = Enumerable.Range(0, trainIndices.Length)
        .Where(i => !foldValidationIndices.Contains(i))
        .ToArray();

    // Create fold data splits
    // Train and evaluate
    var classifier = createClassifier();
    classifier.Train(xFoldTrain, yFoldTrain);
    var predictions = classifier.Predict(xFoldVal);

    // Calculate accuracy for this fold
    foldScores[fold] = correct / total;
}
```

### Calculating Mean and Standard Deviation

```csharp
double mean = foldScores.Average();
double variance = foldScores.Select(s => Math.Pow(s - mean, 2)).Average();
double std = Math.Sqrt(variance);
```

### Building Confusion Matrix

```csharp
var confusionMatrix = new int[3, 3];
for (int i = 0; i < predictions.Length; i++)
{
    int predicted = (int)Math.Round(predictions[i]);
    int actual = (int)yTest[i];
    confusionMatrix[actual, predicted]++;
}
```

## Model Selection Tips

### Choose Random Forest when:
- You need a good baseline quickly
- Interpretability (feature importance) is important
- You want to avoid overfitting

### Choose Gradient Boosting when:
- You need the best possible accuracy
- You have time for hyperparameter tuning
- The dataset isn't too large (slower training)

### Choose SVM when:
- You have high-dimensional data
- You need a memory-efficient model
- You have clear class separation

## Next Steps

- [SentimentAnalysis](../../BinaryClassification/SentimentAnalysis/) - Binary classification with text
- [SpamDetection](../../BinaryClassification/SpamDetection/) - Binary classification with SVM
- [BasicClassification](../../../getting-started/BasicClassification/) - Simpler Iris example
