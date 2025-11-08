# Junior Developer Implementation Guide: Basic Fit Detection (Issue #343)

## Table of Contents
1. [For Beginners: What Is Fit Detection?](#for-beginners-what-is-fit-detection)
2. [What EXISTS in src/FitDetectors/](#what-exists-in-srcfitdetectors)
3. [What's MISSING (Test Coverage)](#whats-missing-test-coverage)
4. [Step-by-Step Test Implementation](#step-by-step-test-implementation)
5. [Testing Strategy](#testing-strategy)
6. [Common Pitfalls](#common-pitfalls)

---

## For Beginners: What Is Fit Detection?

### The Problem
When you train a machine learning model, you need to know if it's learning properly. Three main problems can occur:

1. **Overfitting**: The model memorizes training data instead of learning patterns
   - Like a student memorizing test answers without understanding concepts
   - Performs great on training data but fails on new data

2. **Underfitting**: The model is too simple to capture important patterns
   - Like using a straight line to describe a curve
   - Performs poorly on both training and new data

3. **Good Fit**: The model captures patterns without memorizing noise
   - Like understanding concepts, not just memorizing
   - Performs well on both training and new data

### How FitDetectors Work
FitDetectors analyze model performance metrics (like R-squared, MSE) across three datasets:
- **Training Set**: Data used to train the model
- **Validation Set**: Data used to tune the model
- **Test Set**: Completely new data to evaluate final performance

By comparing these metrics, FitDetectors determine the fit type and provide recommendations.

---

## What EXISTS in src/FitDetectors/

### 1. DefaultFitDetector
**Location**: `C:\Users\cheat\source\repos\AiDotNet\src\FitDetectors\DefaultFitDetector.cs`

**What it does**:
- Simplest fit detector with predefined thresholds
- Compares R-squared values across training/validation/test sets
- Uses fixed thresholds: 0.9 (good), 0.7 (acceptable), 0.5 (poor), 0.2 (variance)

**Key Logic**:
```csharp
// Good fit: All R2 > 0.9
if (training.R2 > 0.9 && validation.R2 > 0.9 && test.R2 > 0.9)
    return FitType.GoodFit;

// Overfit: High training (>0.9), low validation (<0.7)
if (training.R2 > 0.9 && validation.R2 < 0.7)
    return FitType.Overfit;

// Underfit: Both training and validation < 0.7
if (training.R2 < 0.7 && validation.R2 < 0.7)
    return FitType.Underfit;
```

**Confidence Calculation**: Average R2 across all three sets

---

### 2. HoldoutValidationFitDetector
**Location**: `C:\Users\cheat\source\repos\AiDotNet\src\FitDetectors\HoldoutValidationFitDetector.cs`

**What it does**:
- Uses configurable thresholds (via `HoldoutValidationFitDetectorOptions`)
- Compares both R2 and MSE metrics
- Checks stability between validation and test sets

**Key Logic**:
```csharp
// Overfit: Training R2 - Validation R2 > threshold
if (trainingR2 - validationR2 > OverfitThreshold)
    return FitType.Overfit;

// Underfit: Validation R2 < threshold
if (validationR2 < UnderfitThreshold)
    return FitType.Underfit;

// High Variance: |Validation MSE - Test MSE| > threshold
if (|validationMSE - testMSE| > HighVarianceThreshold)
    return FitType.HighVariance;
```

**Confidence Calculation**:
```csharp
1 - (|validationR2 - testR2| / max(validationR2, testR2))
```

---

### 3. KFoldCrossValidationFitDetector
**Location**: `C:\Users\cheat\source\repos\AiDotNet\src\FitDetectors\KFoldCrossValidationFitDetector.cs`

**What it does**:
- Uses K-Fold Cross-Validation approach (data split into K parts)
- Analyzes consistency across different data splits
- Uses average metrics from multiple folds

**Key Logic**: Similar to HoldoutValidationFitDetector but uses averaged metrics

**Confidence Calculation**: Based on consistency between validation and test R2

---

### 4. CrossValidationFitDetector
**Location**: `C:\Users\cheat\source\repos\AiDotNet\src\FitDetectors\CrossValidationFitDetector.cs`

**What it does**: (Need to read this file to confirm - similar to KFold but may use different CV strategy)

---

### 5. BootstrapFitDetector
**Location**: `C:\Users\cheat\source\repos\AiDotNet\src\FitDetectors\BootstrapFitDetector.cs`

**What it does**:
- Uses bootstrap resampling (random sampling with replacement)
- Creates multiple versions of dataset to estimate variability
- Simulates R2 by adding random noise (-0.05 to +0.05)

**Key Logic**:
```csharp
// Performs 1000 bootstrap samples (configurable)
// For each sample, adds noise to R2 values
var noise = random.NextDouble() * 0.1 - 0.05;
var resampledR2 = originalR2 + noise;

// Analyzes distribution of resampled R2 values
// Good fit: Mean R2 > threshold for all sets
// Overfit: Mean training R2 - Mean validation R2 > threshold
```

**Confidence Calculation**: Based on confidence interval width of R2 differences

---

### 6. JackknifeFitDetector
**Location**: `C:\Users\cheat\source\repos\AiDotNet\src\FitDetectors\JackknifeFitDetector.cs`

**What it does**: (Need to read this file to confirm - likely leave-one-out resampling)

---

## What's MISSING (Test Coverage)

### Current State
- **0% test coverage** for all Basic FitDetectors
- No unit tests exist in `tests/` directory
- No integration tests for fit detection workflows

### Required Test Coverage

#### 1. Unit Tests for Each Detector
- Test each FitType detection scenario
- Test edge cases (null values, empty datasets, extreme values)
- Test confidence level calculations
- Test recommendation generation

#### 2. Mock Scenarios
- **Good Fit Scenario**: R2 = 0.95 for all sets
- **Overfit Scenario**: Training R2 = 0.98, Validation R2 = 0.55
- **Underfit Scenario**: All R2 = 0.40
- **High Variance Scenario**: Training R2 = 0.85, Validation R2 = 0.83, Test R2 = 0.45
- **Unstable Scenario**: Mixed metrics that don't fit other categories

#### 3. Edge Cases
- All R2 values at exactly threshold boundaries
- Negative R2 values (worse than baseline)
- R2 = 1.0 (perfect fit)
- Missing validation or test sets
- Null model or evaluation data

---

## Step-by-Step Test Implementation

### Prerequisites
- Install xUnit testing framework
- Reference AiDotNet project
- Create test project structure: `tests/FitDetectors/BasicFitDetectorTests/`

### Step 1: Create Test Helper Class

Create `C:\Users\cheat\source\repos\AiDotNet\tests\FitDetectors\FitDetectorTestHelpers.cs`:

```csharp
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.Tests.FitDetectors;

public static class FitDetectorTestHelpers
{
    /// <summary>
    /// Creates mock ModelEvaluationData with specified R2 values for testing.
    /// </summary>
    public static ModelEvaluationData<double, double[], double> CreateMockEvaluationData(
        double trainingR2,
        double validationR2,
        double testR2,
        double trainingMSE = 0.1,
        double validationMSE = 0.1,
        double testMSE = 0.1)
    {
        return new ModelEvaluationData<double, double[], double>
        {
            TrainingSet = new DataSetStats<double, double[], double>
            {
                PredictionStats = new PredictionStats<double> { R2 = trainingR2 },
                ErrorStats = new ErrorStats<double> { MSE = trainingMSE }
            },
            ValidationSet = new DataSetStats<double, double[], double>
            {
                PredictionStats = new PredictionStats<double> { R2 = validationR2 },
                ErrorStats = new ErrorStats<double> { MSE = validationMSE }
            },
            TestSet = new DataSetStats<double, double[], double>
            {
                PredictionStats = new PredictionStats<double> { R2 = testR2 },
                ErrorStats = new ErrorStats<double> { MSE = testMSE }
            },
            ModelStats = ModelStats<double, double[], double>.Empty()
        };
    }

    /// <summary>
    /// Asserts that FitDetectorResult has expected properties.
    /// </summary>
    public static void AssertFitDetectorResult(
        FitDetectorResult<double> result,
        FitType expectedFitType,
        double? minConfidence = null,
        int? minRecommendations = null)
    {
        Assert.NotNull(result);
        Assert.Equal(expectedFitType, result.FitType);

        if (minConfidence.HasValue)
        {
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= minConfidence.Value,
                $"Expected confidence >= {minConfidence}, got {result.ConfidenceLevel}");
        }

        if (minRecommendations.HasValue)
        {
            Assert.NotNull(result.Recommendations);
            Assert.True(result.Recommendations.Count >= minRecommendations.Value,
                $"Expected at least {minRecommendations} recommendations, got {result.Recommendations.Count}");
        }
    }
}
```

---

### Step 2: DefaultFitDetector Tests

Create `C:\Users\cheat\source\repos\AiDotNet\tests\FitDetectors\DefaultFitDetectorTests.cs`:

```csharp
using AiDotNet.Enums;
using AiDotNet.FitDetectors;
using Xunit;

namespace AiDotNet.Tests.FitDetectors;

public class DefaultFitDetectorTests
{
    private readonly DefaultFitDetector<double, double[], double> _detector;

    public DefaultFitDetectorTests()
    {
        _detector = new DefaultFitDetector<double, double[], double>();
    }

    [Fact]
    public void DetectFit_GoodFit_AllR2Above90()
    {
        // Arrange: All R2 values > 0.9
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(
            trainingR2: 0.95,
            validationR2: 0.93,
            testR2: 0.92);

        // Act
        var result = _detector.DetectFit(evalData);

        // Assert
        FitDetectorTestHelpers.AssertFitDetectorResult(
            result,
            expectedFitType: FitType.GoodFit,
            minConfidence: 0.9,
            minRecommendations: 1);
    }

    [Fact]
    public void DetectFit_Overfit_HighTrainingLowValidation()
    {
        // Arrange: Training R2 > 0.9, Validation R2 < 0.7
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(
            trainingR2: 0.98,
            validationR2: 0.55,
            testR2: 0.50);

        // Act
        var result = _detector.DetectFit(evalData);

        // Assert
        FitDetectorTestHelpers.AssertFitDetectorResult(
            result,
            expectedFitType: FitType.Overfit,
            minRecommendations: 2);

        // Verify recommendations include regularization
        Assert.Contains(result.Recommendations,
            r => r.Contains("regularization", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void DetectFit_Underfit_BothLow()
    {
        // Arrange: Both training and validation R2 < 0.7
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(
            trainingR2: 0.45,
            validationR2: 0.40,
            testR2: 0.38);

        // Act
        var result = _detector.DetectFit(evalData);

        // Assert
        FitDetectorTestHelpers.AssertFitDetectorResult(
            result,
            expectedFitType: FitType.Underfit,
            minRecommendations: 2);

        // Verify recommendations include increasing complexity
        Assert.Contains(result.Recommendations,
            r => r.Contains("complexity", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void DetectFit_HighVariance_LargeDifferenceBetweenTrainingAndValidation()
    {
        // Arrange: |Training R2 - Validation R2| > 0.2
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(
            trainingR2: 0.85,
            validationR2: 0.60,
            testR2: 0.58);

        // Act
        var result = _detector.DetectFit(evalData);

        // Assert
        FitDetectorTestHelpers.AssertFitDetectorResult(
            result,
            expectedFitType: FitType.HighVariance,
            minRecommendations: 2);
    }

    [Fact]
    public void DetectFit_HighBias_AllBelow50()
    {
        // Arrange: All R2 < 0.5
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(
            trainingR2: 0.30,
            validationR2: 0.28,
            testR2: 0.25);

        // Act
        var result = _detector.DetectFit(evalData);

        // Assert
        FitDetectorTestHelpers.AssertFitDetectorResult(
            result,
            expectedFitType: FitType.HighBias,
            minRecommendations: 2);
    }

    [Fact]
    public void DetectFit_Unstable_NoOtherCategoryMatches()
    {
        // Arrange: Mixed metrics that don't fit other categories
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(
            trainingR2: 0.75,
            validationR2: 0.72,
            testR2: 0.68);

        // Act
        var result = _detector.DetectFit(evalData);

        // Assert
        FitDetectorTestHelpers.AssertFitDetectorResult(
            result,
            expectedFitType: FitType.Unstable,
            minRecommendations: 2);
    }

    [Fact]
    public void CalculateConfidenceLevel_ReturnsAverageR2()
    {
        // Arrange
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(
            trainingR2: 0.90,
            validationR2: 0.80,
            testR2: 0.70);

        // Act
        var result = _detector.DetectFit(evalData);

        // Assert: Confidence = (0.90 + 0.80 + 0.70) / 3 = 0.80
        Assert.NotNull(result.ConfidenceLevel);
        Assert.Equal(0.80, result.ConfidenceLevel.Value, precision: 2);
    }

    [Fact]
    public void DetectFit_EdgeCase_PerfectFit()
    {
        // Arrange: R2 = 1.0 (perfect fit)
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(
            trainingR2: 1.0,
            validationR2: 1.0,
            testR2: 1.0);

        // Act
        var result = _detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.GoodFit, result.FitType);
        Assert.Equal(1.0, result.ConfidenceLevel);
    }

    [Fact]
    public void DetectFit_EdgeCase_NegativeR2()
    {
        // Arrange: Negative R2 (worse than baseline)
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(
            trainingR2: -0.5,
            validationR2: -0.3,
            testR2: -0.4);

        // Act
        var result = _detector.DetectFit(evalData);

        // Assert: Should detect as HighBias (all very low)
        Assert.Equal(FitType.HighBias, result.FitType);
    }

    [Fact]
    public void DetectFit_BoundaryCase_ExactlyAtThreshold()
    {
        // Arrange: R2 exactly at 0.7 threshold
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(
            trainingR2: 0.70,
            validationR2: 0.70,
            testR2: 0.70);

        // Act
        var result = _detector.DetectFit(evalData);

        // Assert: Should NOT be Underfit (requires < 0.7)
        Assert.NotEqual(FitType.Underfit, result.FitType);
    }
}
```

---

### Step 3: HoldoutValidationFitDetector Tests

Create `C:\Users\cheat\source\repos\AiDotNet\tests\FitDetectors\HoldoutValidationFitDetectorTests.cs`:

```csharp
using AiDotNet.Enums;
using AiDotNet.FitDetectors;
using AiDotNet.Options.FitDetectors;
using Xunit;

namespace AiDotNet.Tests.FitDetectors;

public class HoldoutValidationFitDetectorTests
{
    [Fact]
    public void DetectFit_UsesDefaultOptions_WhenNoneProvided()
    {
        // Arrange
        var detector = new HoldoutValidationFitDetector<double, double[], double>();
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(0.95, 0.92, 0.90);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.NotNull(result);
    }

    [Fact]
    public void DetectFit_UsesCustomOptions()
    {
        // Arrange: Custom thresholds
        var options = new HoldoutValidationFitDetectorOptions
        {
            OverfitThreshold = 0.15,
            UnderfitThreshold = 0.60,
            GoodFitThreshold = 0.85,
            HighVarianceThreshold = 0.10,
            StabilityThreshold = 0.05
        };
        var detector = new HoldoutValidationFitDetector<double, double[], double>(options);

        // Training - Validation = 0.85 - 0.72 = 0.13 < 0.15 (not overfit with custom threshold)
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(0.85, 0.72, 0.70);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: Should NOT be Overfit due to higher threshold
        Assert.NotEqual(FitType.Overfit, result.FitType);
    }

    [Fact]
    public void DetectFit_Overfit_BasedOnR2Difference()
    {
        // Arrange
        var detector = new HoldoutValidationFitDetector<double, double[], double>();
        // Default OverfitThreshold is typically 0.1
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(
            trainingR2: 0.90,
            validationR2: 0.70,  // Difference = 0.20 > 0.1
            testR2: 0.68);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.Overfit, result.FitType);
    }

    [Fact]
    public void DetectFit_HighVariance_BasedOnMSEDifference()
    {
        // Arrange
        var detector = new HoldoutValidationFitDetector<double, double[], double>();
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(
            trainingR2: 0.85,
            validationR2: 0.83,
            testR2: 0.80,
            validationMSE: 0.10,
            testMSE: 0.25);  // Large MSE difference

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.HighVariance, result.FitType);
    }

    [Fact]
    public void CalculateConfidenceLevel_BasedOnValidationTestConsistency()
    {
        // Arrange
        var detector = new HoldoutValidationFitDetector<double, double[], double>();

        // Case 1: Very consistent validation and test
        var evalData1 = FitDetectorTestHelpers.CreateMockEvaluationData(
            trainingR2: 0.90,
            validationR2: 0.88,
            testR2: 0.87);  // Difference = 0.01

        // Case 2: Inconsistent validation and test
        var evalData2 = FitDetectorTestHelpers.CreateMockEvaluationData(
            trainingR2: 0.90,
            validationR2: 0.85,
            testR2: 0.55);  // Difference = 0.30

        // Act
        var result1 = detector.DetectFit(evalData1);
        var result2 = detector.DetectFit(evalData2);

        // Assert: Consistent should have higher confidence
        Assert.True(result1.ConfidenceLevel > result2.ConfidenceLevel);
    }

    [Fact]
    public void GenerateRecommendations_IncludesR2Values()
    {
        // Arrange
        var detector = new HoldoutValidationFitDetector<double, double[], double>();
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(0.95, 0.92, 0.90);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: Should include R2 values in recommendations
        var r2Recommendation = result.Recommendations.FirstOrDefault(
            r => r.Contains("R2") || r.Contains("r2"));
        Assert.NotNull(r2Recommendation);
    }
}
```

---

### Step 4: KFoldCrossValidationFitDetector Tests

Similar pattern to HoldoutValidationFitDetector, focusing on:
- Default K value (typically 5 or 10)
- Custom K values
- Averaged metrics across folds
- Consistency between folds

---

### Step 5: BootstrapFitDetector Tests

Create `C:\Users\cheat\source\repos\AiDotNet\tests\FitDetectors\BootstrapFitDetectorTests.cs`:

```csharp
using AiDotNet.Enums;
using AiDotNet.FitDetectors;
using AiDotNet.Options.FitDetectors;
using Xunit;

namespace AiDotNet.Tests.FitDetectors;

public class BootstrapFitDetectorTests
{
    [Fact]
    public void DetectFit_UsesConfiguredNumberOfBootstraps()
    {
        // Arrange
        var options = new BootstrapFitDetectorOptions
        {
            NumberOfBootstraps = 100  // Lower for faster testing
        };
        var detector = new BootstrapFitDetector<double, double[], double>(options);
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(0.90, 0.88, 0.85);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.NotNull(result);
        // Check that recommendations mention bootstrap count
        var bootstrapMention = result.Recommendations.FirstOrDefault(
            r => r.Contains("100") && r.Contains("bootstrap", StringComparison.OrdinalIgnoreCase));
        Assert.NotNull(bootstrapMention);
    }

    [Fact]
    public void DetectFit_GoodFit_WhenAllMeanR2High()
    {
        // Arrange
        var detector = new BootstrapFitDetector<double, double[], double>();
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(
            trainingR2: 0.95,
            validationR2: 0.93,
            testR2: 0.91);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.GoodFit, result.FitType);
    }

    [Fact]
    public void DetectFit_Overfit_WhenBootstrapMeanShowsLargeGap()
    {
        // Arrange
        var detector = new BootstrapFitDetector<double, double[], double>();
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(
            trainingR2: 0.98,
            validationR2: 0.60,
            testR2: 0.58);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.Overfit, result.FitType);
    }

    [Fact]
    public void CalculateConfidenceLevel_NarrowConfidenceInterval_HighConfidence()
    {
        // Arrange
        var detector = new BootstrapFitDetector<double, double[], double>();

        // Very consistent metrics = narrow confidence interval
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(
            trainingR2: 0.90,
            validationR2: 0.89,
            testR2: 0.88);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: Should have high confidence (narrow interval)
        Assert.NotNull(result.ConfidenceLevel);
        Assert.True(result.ConfidenceLevel > 0.5);
    }

    [Fact]
    public void DetectFit_Underfit_WhenAllBootstrapMeansLow()
    {
        // Arrange
        var detector = new BootstrapFitDetector<double, double[], double>();
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(
            trainingR2: 0.35,
            validationR2: 0.30,
            testR2: 0.28);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.Underfit, result.FitType);
    }

    [Fact]
    public void DetectFit_RepeatedRuns_ProduceConsistentResults()
    {
        // Arrange: Bootstrap uses randomness, but should be stable for clear cases
        var detector = new BootstrapFitDetector<double, double[], double>();
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(0.95, 0.93, 0.91);

        // Act: Run multiple times
        var result1 = detector.DetectFit(evalData);
        var result2 = detector.DetectFit(evalData);
        var result3 = detector.DetectFit(evalData);

        // Assert: FitType should be consistent (even if confidence varies slightly)
        Assert.Equal(result1.FitType, result2.FitType);
        Assert.Equal(result2.FitType, result3.FitType);
    }
}
```

---

### Step 6: Integration Tests

Create `C:\Users\cheat\source\repos\AiDotNet\tests\FitDetectors\FitDetectorIntegrationTests.cs`:

```csharp
using AiDotNet.Enums;
using AiDotNet.FitDetectors;
using Xunit;

namespace AiDotNet.Tests.FitDetectors;

public class FitDetectorIntegrationTests
{
    [Theory]
    [InlineData(0.95, 0.93, 0.91, FitType.GoodFit)]
    [InlineData(0.98, 0.55, 0.52, FitType.Overfit)]
    [InlineData(0.40, 0.38, 0.35, FitType.Underfit)]
    public void AllBasicDetectors_DetectSameScenario_ConsistentResults(
        double trainingR2,
        double validationR2,
        double testR2,
        FitType expectedFitType)
    {
        // Arrange: Create same evaluation data for all detectors
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(
            trainingR2, validationR2, testR2);

        var defaultDetector = new DefaultFitDetector<double, double[], double>();
        var holdoutDetector = new HoldoutValidationFitDetector<double, double[], double>();
        var kfoldDetector = new KFoldCrossValidationFitDetector<double, double[], double>();
        var bootstrapDetector = new BootstrapFitDetector<double, double[], double>();

        // Act
        var defaultResult = defaultDetector.DetectFit(evalData);
        var holdoutResult = holdoutDetector.DetectFit(evalData);
        var kfoldResult = kfoldDetector.DetectFit(evalData);
        var bootstrapResult = bootstrapDetector.DetectFit(evalData);

        // Assert: All should detect the same FitType for clear cases
        Assert.Equal(expectedFitType, defaultResult.FitType);
        Assert.Equal(expectedFitType, holdoutResult.FitType);
        Assert.Equal(expectedFitType, kfoldResult.FitType);
        Assert.Equal(expectedFitType, bootstrapResult.FitType);
    }

    [Fact]
    public void AllDetectors_ProduceRecommendations()
    {
        // Arrange
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(0.98, 0.55, 0.52);

        var detectors = new IFitDetector<double, double[], double>[]
        {
            new DefaultFitDetector<double, double[], double>(),
            new HoldoutValidationFitDetector<double, double[], double>(),
            new KFoldCrossValidationFitDetector<double, double[], double>(),
            new BootstrapFitDetector<double, double[], double>()
        };

        // Act & Assert
        foreach (var detector in detectors)
        {
            var result = detector.DetectFit(evalData);
            Assert.NotNull(result.Recommendations);
            Assert.NotEmpty(result.Recommendations);
        }
    }
}
```

---

## Testing Strategy

### 1. Test Pyramid
- **70% Unit Tests**: Test individual detector logic, thresholds, calculations
- **20% Integration Tests**: Test detectors on same scenarios, consistency
- **10% Edge Case Tests**: Boundary values, null handling, extreme inputs

### 2. Test Data Strategy
- Use `FitDetectorTestHelpers.CreateMockEvaluationData()` for consistency
- Create named test data sets (e.g., `GoodFitData`, `OverfitData`)
- Use Theory tests with InlineData for multiple scenarios

### 3. Assertion Strategy
- Use `FitDetectorTestHelpers.AssertFitDetectorResult()` for common checks
- Verify FitType, ConfidenceLevel, and Recommendations separately
- Check recommendation content, not just count

### 4. Coverage Goals
- **Minimum**: 80% code coverage for each detector
- **Target**: 90% code coverage including edge cases
- **Focus**: All public methods and all FitType branches

---

## Common Pitfalls

### 1. Floating-Point Precision
**Problem**: Comparing doubles with `==` fails due to precision issues

**Wrong**:
```csharp
Assert.Equal(0.80, result.ConfidenceLevel);
```

**Correct**:
```csharp
Assert.Equal(0.80, result.ConfidenceLevel.Value, precision: 2);
```

---

### 2. Null Reference Exceptions
**Problem**: Not checking for null before accessing properties

**Wrong**:
```csharp
Assert.True(result.ConfidenceLevel >= 0.5);  // NullReferenceException if null
```

**Correct**:
```csharp
Assert.NotNull(result.ConfidenceLevel);
Assert.True(result.ConfidenceLevel.Value >= 0.5);
```

---

### 3. String Comparison Case Sensitivity
**Problem**: Recommendation checks fail due to case differences

**Wrong**:
```csharp
Assert.Contains("Regularization", result.Recommendations[0]);
```

**Correct**:
```csharp
Assert.Contains(result.Recommendations,
    r => r.Contains("regularization", StringComparison.OrdinalIgnoreCase));
```

---

### 4. Threshold Boundary Testing
**Problem**: Not testing values exactly at thresholds

**Example**: If threshold is 0.7, test these cases:
- R2 = 0.69 (below threshold)
- R2 = 0.70 (exactly at threshold)
- R2 = 0.71 (above threshold)

---

### 5. Bootstrap/Random Behavior
**Problem**: Tests fail intermittently due to randomness

**Solution**:
- Test for FitType consistency across runs (not exact confidence values)
- Use higher bootstrap counts for stability
- Test ranges instead of exact values for confidence

**Example**:
```csharp
// Don't test exact confidence value
Assert.Equal(0.85, result.ConfidenceLevel);  // May vary

// Test confidence range
Assert.InRange(result.ConfidenceLevel.Value, 0.80, 0.90);
```

---

### 6. Missing Test Scenarios
**Common Missing Tests**:
- Negative R2 values (worse than baseline)
- R2 = 1.0 (perfect fit)
- R2 = 0.0 (no predictive power)
- All three sets having identical values
- Large gaps between consecutive sets

---

### 7. Generic Type Handling
**Problem**: Tests only use `double`, not testing generic nature

**Solution**: Create tests for both `float` and `double`:
```csharp
[Theory]
[InlineData(typeof(double))]
[InlineData(typeof(float))]
public void DetectFit_WorksWithDifferentNumericTypes(Type numericType)
{
    // Test generic type T behavior
}
```

---

## Next Steps

1. **Create test project**: `tests/AiDotNet.Tests.csproj`
2. **Implement helper class**: `FitDetectorTestHelpers.cs`
3. **Write tests for each detector**:
   - DefaultFitDetector
   - HoldoutValidationFitDetector
   - KFoldCrossValidationFitDetector
   - CrossValidationFitDetector
   - BootstrapFitDetector
   - JackknifeFitDetector
4. **Run tests and verify coverage**: `dotnet test --collect:"XPlat Code Coverage"`
5. **Fix failing tests and add missing scenarios**
6. **Document test results in PR**

---

## Additional Resources

- **FitDetector Interface**: `src/Interfaces/IFitDetector.cs`
- **FitDetectorBase**: `src/FitDetectors/FitDetectorBase.cs`
- **FitType Enum**: `src/Enums/FitType.cs`
- **ModelEvaluationData**: `src/Models/ModelEvaluationData.cs`
- **FitDetectorResult**: `src/Models/Results/FitDetectorResult.cs`

---

## Questions?

If you encounter issues:
1. Check existing detector implementations for patterns
2. Review FitDetectorBase for inherited behavior
3. Look at FitType enum for all possible return values
4. Verify your mock data matches expected structure
5. Ask for help in PR comments or issue discussions
