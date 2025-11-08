# Issue #351: Junior Developer Implementation Guide
## Unit Tests for Data Processing Helpers

---

## Overview

Create comprehensive unit tests for data processing helper classes in `src/Helpers/` that prepare and transform data for AI models.

**Files to Test**:
- `InputHelper.cs` - Input data processing
- `FeatureSelectorHelper.cs` - Feature selection algorithms
- `OutlierRemovalHelper.cs` - Outlier detection and removal
- `SamplingHelper.cs` - Data sampling strategies
- `AdaptiveParametersHelper.cs` - Adaptive parameter tuning

**Target**: 0% → 80% test coverage

---

## InputHelper Testing

### Key Methods to Test

1. **NormalizeInput()** - Normalize input data
2. **StandardizeInput()** - Standardize (z-score normalization)
3. **ValidateInput()** - Check input validity
4. **ConvertToMatrix()** - Convert various formats to Matrix<T>
5. **ConvertToTensor()** - Convert various formats to Tensor<T>
6. **SplitData()** - Split into train/validation/test sets
7. **ShuffleData()** - Randomly shuffle dataset
8. **PadSequences()** - Pad variable-length sequences

### Test Examples

```csharp
[TestClass]
public class InputHelperTests
{
    [TestMethod]
    public void NormalizeInput_ToZeroOne_ScalesCorrectly()
    {
        // Arrange
        var data = new Vector<double>(new[] { 0.0, 5.0, 10.0 });

        // Act
        var normalized = InputHelper.NormalizeInput(data, min: 0.0, max: 1.0);

        // Assert
        Assert.AreEqual(0.0, normalized[0], 1e-10);
        Assert.AreEqual(0.5, normalized[1], 1e-10);
        Assert.AreEqual(1.0, normalized[2], 1e-10);
    }

    [TestMethod]
    public void StandardizeInput_WithMeanZeroStdOne_ProducesZScores()
    {
        // Arrange
        var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        // Mean = 3.0, StdDev = sqrt(2)

        // Act
        var standardized = InputHelper.StandardizeInput(data);

        // Assert
        double mean = standardized.Average();
        double stdDev = Math.Sqrt(standardized.Select(x => x * x).Average());

        Assert.IsTrue(Math.Abs(mean) < 1e-10, "Mean should be ~0");
        Assert.IsTrue(Math.Abs(stdDev - 1.0) < 0.1, "StdDev should be ~1");
    }

    [TestMethod]
    public void SplitData_With70_30_CreatesCorrectSplits()
    {
        // Arrange
        var X = new Matrix<double>(100, 5);
        var y = new Vector<double>(100);

        // Act
        var (XTrain, yTrain, XTest, yTest) = InputHelper.SplitData(
            X, y, trainRatio: 0.7);

        // Assert
        Assert.AreEqual(70, XTrain.Rows);
        Assert.AreEqual(70, yTrain.Length);
        Assert.AreEqual(30, XTest.Rows);
        Assert.AreEqual(30, yTest.Length);
    }
}
```

### Test Checklist
- [ ] Normalization (min-max scaling)
- [ ] Standardization (z-score)
- [ ] Input validation (null, empty, dimension mismatch)
- [ ] Data splitting (various ratios)
- [ ] Shuffling (verify randomness)
- [ ] Sequence padding (variable lengths)
- [ ] Format conversions

---

## FeatureSelectorHelper Testing

### Key Methods to Test

1. **SelectByVariance()** - Remove low-variance features
2. **SelectByCorrelation()** - Remove highly correlated features
3. **SelectByImportance()** - Select top-k important features
4. **SelectByForwardSelection()** - Forward feature selection
5. **SelectByBackwardElimination()** - Backward feature elimination
6. **SelectByGeneticAlgorithm()** - GA-based selection
7. **CalculateFeatureImportance()** - Compute importance scores

### Test Examples

```csharp
[TestClass]
public class FeatureSelectorHelperTests
{
    [TestMethod]
    public void SelectByVariance_RemovesConstantFeatures()
    {
        // Arrange
        var X = new Matrix<double>(10, 3);
        // Feature 0: constant (variance = 0)
        for (int i = 0; i < 10; i++)
        {
            X[i, 0] = 5.0;
            X[i, 1] = i;        // Varying feature
            X[i, 2] = i * 2;    // Varying feature
        }

        // Act
        var selected = FeatureSelectorHelper.SelectByVariance(
            X, minVariance: 0.1);

        // Assert
        Assert.AreEqual(2, selected.Count, "Should select 2 varying features");
        Assert.IsFalse(selected.Contains(0), "Constant feature should be removed");
        Assert.IsTrue(selected.Contains(1));
        Assert.IsTrue(selected.Contains(2));
    }

    [TestMethod]
    public void SelectByCorrelation_RemovesHighlyCorrelatedFeatures()
    {
        // Arrange
        var X = new Matrix<double>(10, 3);
        for (int i = 0; i < 10; i++)
        {
            X[i, 0] = i;
            X[i, 1] = i * 2;    // Perfectly correlated with feature 0
            X[i, 2] = -i;       // Negatively correlated with feature 0
        }

        // Act
        var selected = FeatureSelectorHelper.SelectByCorrelation(
            X, maxCorrelation: 0.95);

        // Assert
        Assert.IsTrue(selected.Count < 3,
            "Should remove some highly correlated features");
    }

    [TestMethod]
    public void SelectByImportance_SelectsTopKFeatures()
    {
        // Arrange
        var importances = new Dictionary<int, double>
        {
            { 0, 0.1 },
            { 1, 0.5 },
            { 2, 0.3 },
            { 3, 0.8 },
            { 4, 0.2 }
        };

        // Act
        var selected = FeatureSelectorHelper.SelectByImportance(
            importances, topK: 2);

        // Assert
        Assert.AreEqual(2, selected.Count);
        Assert.IsTrue(selected.Contains(3), "Should include feature 3 (0.8)");
        Assert.IsTrue(selected.Contains(1), "Should include feature 1 (0.5)");
    }
}
```

### Test Checklist
- [ ] Variance threshold selection
- [ ] Correlation-based removal
- [ ] Importance-based selection
- [ ] Forward selection algorithm
- [ ] Backward elimination algorithm
- [ ] GA-based feature selection
- [ ] Feature importance calculation
- [ ] Edge cases (all features selected/removed)

---

## OutlierRemovalHelper Testing

### Key Methods to Test

1. **DetectOutliersByZScore()** - Z-score based detection
2. **DetectOutliersByIQR()** - Interquartile range detection
3. **DetectOutliersByIsolationForest()** - Isolation forest
4. **RemoveOutliers()** - Remove detected outliers
5. **ReplaceOutliers()** - Replace with mean/median
6. **CalculateOutlierScore()** - Score each sample

### Test Examples

```csharp
[TestClass]
public class OutlierRemovalHelperTests
{
    [TestMethod]
    public void DetectOutliersByZScore_IdentifiesExtremeValues()
    {
        // Arrange
        var data = new Vector<double>(new[]
        {
            1.0, 2.0, 3.0, 4.0, 5.0,  // Normal values
            100.0                      // Outlier
        });

        // Act
        var outliers = OutlierRemovalHelper.DetectOutliersByZScore(
            data, threshold: 3.0);

        // Assert
        Assert.AreEqual(1, outliers.Count);
        Assert.IsTrue(outliers.Contains(5), "Index 5 (value 100) is outlier");
    }

    [TestMethod]
    public void DetectOutliersByIQR_UsesQuartiles()
    {
        // Arrange
        var data = new Vector<double>(new[]
        {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
            50.0  // Outlier beyond 1.5*IQR
        });

        // Act
        var outliers = OutlierRemovalHelper.DetectOutliersByIQR(data);

        // Assert
        Assert.AreEqual(1, outliers.Count);
        Assert.IsTrue(outliers.Contains(9), "Index 9 (value 50) is outlier");
    }

    [TestMethod]
    public void RemoveOutliers_RemovesDetectedIndices()
    {
        // Arrange
        var X = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);
        var outlierIndices = new List<int> { 2, 4 };

        // Act
        var (XClean, yClean) = OutlierRemovalHelper.RemoveOutliers(
            X, y, outlierIndices);

        // Assert
        Assert.AreEqual(4, XClean.Rows);
        Assert.AreEqual(4, yClean.Length);
    }
}
```

### Test Checklist
- [ ] Z-score detection (various thresholds)
- [ ] IQR detection
- [ ] Isolation forest detection
- [ ] Outlier removal
- [ ] Outlier replacement (mean, median)
- [ ] Scoring mechanism
- [ ] Edge cases (no outliers, all outliers)

---

## SamplingHelper Testing

### Key Methods to Test

1. **RandomSample()** - Random sampling
2. **StratifiedSample()** - Stratified sampling (preserve class distribution)
3. **BootstrapSample()** - Bootstrap resampling
4. **CrossValidationSplit()** - K-fold cross-validation splits
5. **TimeSeriesSplit()** - Time-aware splitting
6. **OversampleMinority()** - SMOTE or random oversampling
7. **UndersampleMajority()** - Random undersampling

### Test Examples

```csharp
[TestClass]
public class SamplingHelperTests
{
    [TestMethod]
    public void RandomSample_SelectsCorrectCount()
    {
        // Arrange
        var data = new Matrix<double>(100, 5);
        int sampleSize = 30;

        // Act
        var sample = SamplingHelper.RandomSample(data, sampleSize);

        // Assert
        Assert.AreEqual(30, sample.Rows);
    }

    [TestMethod]
    public void StratifiedSample_PreservesClassDistribution()
    {
        // Arrange
        var X = new Matrix<double>(100, 2);
        var y = new Vector<double>(100);
        // 70 samples of class 0, 30 samples of class 1
        for (int i = 0; i < 70; i++) y[i] = 0;
        for (int i = 70; i < 100; i++) y[i] = 1;

        // Act
        var (XSample, ySample) = SamplingHelper.StratifiedSample(
            X, y, sampleSize: 50);

        // Assert
        int class0Count = ySample.Count(label => label == 0);
        int class1Count = ySample.Count(label => label == 1);

        Assert.AreEqual(35, class0Count, "Should have 70% class 0");
        Assert.AreEqual(15, class1Count, "Should have 30% class 1");
    }

    [TestMethod]
    public void BootstrapSample_AllowsDuplicates()
    {
        // Arrange
        var data = new Matrix<double>(10, 2);

        // Act
        var bootstrap = SamplingHelper.BootstrapSample(data, sampleSize: 20);

        // Assert
        Assert.AreEqual(20, bootstrap.Rows,
            "Bootstrap can sample more than original size");
    }

    [TestMethod]
    public void CrossValidationSplit_CreatesKFolds()
    {
        // Arrange
        var X = new Matrix<double>(100, 5);
        var y = new Vector<double>(100);
        int k = 5;

        // Act
        var folds = SamplingHelper.CrossValidationSplit(X, y, k);

        // Assert
        Assert.AreEqual(k, folds.Count);
        Assert.IsTrue(folds.All(fold => fold.TrainX.Rows == 80),
            "Each fold should have 80% training data");
        Assert.IsTrue(folds.All(fold => fold.ValidX.Rows == 20),
            "Each fold should have 20% validation data");
    }
}
```

### Test Checklist
- [ ] Random sampling (various sizes)
- [ ] Stratified sampling (class distribution)
- [ ] Bootstrap sampling (with replacement)
- [ ] K-fold cross-validation
- [ ] Time series splitting
- [ ] Oversampling (SMOTE)
- [ ] Undersampling
- [ ] Edge cases (sample size > data size)

---

## AdaptiveParametersHelper Testing

### Key Methods to Test

1. **AdaptLearningRate()** - Adjust learning rate based on progress
2. **AdaptMomentum()** - Adjust momentum based on gradient variance
3. **AdaptBatchSize()** - Adjust batch size dynamically
4. **AdaptRegularization()** - Tune L1/L2 regularization
5. **DetectPlateau()** - Detect training plateau
6. **SuggestParameterAdjustment()** - Recommend parameter changes

### Test Examples

```csharp
[TestClass]
public class AdaptiveParametersHelperTests
{
    [TestMethod]
    public void AdaptLearningRate_OnPlateau_DecreasesRate()
    {
        // Arrange
        double initialRate = 0.1;
        var lossHistory = new List<double>
        {
            1.0, 0.9, 0.85, 0.84, 0.84, 0.84  // Plateau
        };

        // Act
        double newRate = AdaptiveParametersHelper.AdaptLearningRate(
            currentRate: initialRate,
            lossHistory: lossHistory,
            patience: 3
        );

        // Assert
        Assert.IsTrue(newRate < initialRate,
            "Learning rate should decrease on plateau");
    }

    [TestMethod]
    public void DetectPlateau_WithConstantLoss_ReturnsTrue()
    {
        // Arrange
        var lossHistory = new List<double>
        {
            0.5, 0.5, 0.5, 0.5, 0.5
        };

        // Act
        bool isPlateau = AdaptiveParametersHelper.DetectPlateau(
            lossHistory, tolerance: 0.01, patience: 3);

        // Assert
        Assert.IsTrue(isPlateau);
    }

    [TestMethod]
    public void DetectPlateau_WithDecreasingLoss_ReturnsFalse()
    {
        // Arrange
        var lossHistory = new List<double>
        {
            1.0, 0.8, 0.6, 0.4, 0.2
        };

        // Act
        bool isPlateau = AdaptiveParametersHelper.DetectPlateau(
            lossHistory, tolerance: 0.01, patience: 3);

        // Assert
        Assert.IsFalse(isPlateau);
    }
}
```

### Test Checklist
- [ ] Learning rate adaptation
- [ ] Momentum adaptation
- [ ] Batch size adaptation
- [ ] Regularization tuning
- [ ] Plateau detection
- [ ] Parameter suggestions
- [ ] Edge cases (empty history, unstable training)

---

## Test File Structure

```
tests/Helpers/
├── InputHelperTests.cs
├── FeatureSelectorHelperTests.cs
├── OutlierRemovalHelperTests.cs
├── SamplingHelperTests.cs
└── AdaptiveParametersHelperTests.cs
```

---

## Success Criteria

### Definition of Done

- [ ] 5 test files created
- [ ] 75+ tests total (15 per helper minimum)
- [ ] All tests passing
- [ ] Code coverage >= 75% for each helper
- [ ] Edge cases tested
- [ ] Data integrity verified after transformations

### Quality Checklist

- [ ] All sampling methods tested
- [ ] Feature selection algorithms validated
- [ ] Outlier detection accuracy verified
- [ ] Data splits preserve distributions
- [ ] Adaptive parameters respond to training dynamics
- [ ] Input validation comprehensive

---

## Resources

- See ISSUE_349_JUNIOR_DEV_GUIDE.md for helper testing patterns
- See ISSUE_347_JUNIOR_DEV_GUIDE.md for genetic algorithm feature selection
- Statistical methods reference: scipy.stats documentation

---

**Target**: Create tests ensuring data processing helpers produce valid, clean data for model training.
