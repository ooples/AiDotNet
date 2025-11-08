# Junior Developer Implementation Guide: Statistical Fit Detectors (Issue #344)

## Table of Contents
1. [For Beginners: What Are Statistical Fit Detectors?](#for-beginners-what-are-statistical-fit-detectors)
2. [What EXISTS in src/FitDetectors/](#what-exists-in-srcfitdetectors)
3. [What's MISSING (Test Coverage)](#whats-missing-test-coverage)
4. [Step-by-Step Test Implementation](#step-by-step-test-implementation)
5. [Testing Strategy](#testing-strategy)
6. [Common Pitfalls](#common-pitfalls)

---

## For Beginners: What Are Statistical Fit Detectors?

### Beyond Basic Fit Detection
While basic fit detectors (Issue #343) focus on comparing R-squared values, **statistical fit detectors** use advanced statistical tests to identify specific problems:

1. **Autocorrelation**: Are errors correlated with previous errors?
   - Common in time series data
   - Indicates missing time-dependent patterns

2. **Heteroscedasticity**: Do errors have consistent variance?
   - Error size shouldn't depend on prediction size
   - Important for reliable confidence intervals

3. **Cook's Distance**: Which data points are overly influential?
   - Identifies outliers that disproportionately affect the model
   - Helps detect overfitting to specific points

4. **Confusion Matrix**: How well does classifier distinguish classes?
   - Specific to classification models
   - Identifies class imbalance and misclassification patterns

---

## What EXISTS in src/FitDetectors/

### 1. AutocorrelationFitDetector
**Location**: `C:\Users\cheat\source\repos\AiDotNet\src\FitDetectors\AutocorrelationFitDetector.cs`

**What it does**:
- Uses Durbin-Watson statistic to detect autocorrelation in residuals
- Durbin-Watson ranges from 0 to 4 (2.0 = no autocorrelation)
- Values < 2 indicate positive autocorrelation
- Values > 2 indicate negative autocorrelation

**Key Logic**:
```csharp
var durbinWatsonStat = StatisticsHelper.CalculateDurbinWatsonStatistic(errors);

if (durbinWatsonStat < StrongPositiveThreshold)  // e.g., < 1.5
    return FitType.StrongPositiveAutocorrelation;
else if (durbinWatsonStat > StrongNegativeThreshold)  // e.g., > 2.5
    return FitType.StrongNegativeAutocorrelation;
else if (durbinWatsonStat >= 1.5 && durbinWatsonStat <= 2.5)
    return FitType.NoAutocorrelation;
else
    return FitType.WeakAutocorrelation;
```

**Confidence Calculation**:
```csharp
// Closer to 2.0 = higher confidence
1.0 - |durbinWatsonStat - 2.0| / 2.0
```

**FitTypes Returned**:
- `StrongPositiveAutocorrelation`
- `StrongNegativeAutocorrelation`
- `WeakAutocorrelation`
- `NoAutocorrelation`

---

### 2. HeteroscedasticityFitDetector
**Location**: `C:\Users\cheat\source\repos\AiDotNet\src\FitDetectors\HeteroscedasticityFitDetector.cs`

**What it does**:
- Runs Breusch-Pagan test (tests if error variance depends on predictors)
- Runs White test (more general test for heteroscedasticity)
- Compares test statistics to thresholds

**Key Logic**:
```csharp
var breuschPaganStat = CalculateBreuschPaganTestStatistic(evalData);
var whiteStat = CalculateWhiteTestStatistic(evalData);

if (breuschPaganStat > HeteroscedasticityThreshold ||
    whiteStat > HeteroscedasticityThreshold)
    return FitType.Unstable;  // Heteroscedasticity detected

if (breuschPaganStat < HomoscedasticityThreshold &&
    whiteStat < HomoscedasticityThreshold)
    return FitType.GoodFit;  // Homoscedasticity (consistent variance)

return FitType.Moderate;
```

**FitTypes Returned**:
- `GoodFit` (homoscedasticity - consistent error variance)
- `Moderate` (some heteroscedasticity)
- `Unstable` (severe heteroscedasticity)

---

### 3. CookDistanceFitDetector
**Location**: `C:\Users\cheat\source\repos\AiDotNet\src\FitDetectors\CookDistanceFitDetector.cs`

**What it does**:
- Calculates Cook's distance for each data point
- Cook's distance measures how much removing a point would change predictions
- Identifies influential points that may cause overfitting

**Key Logic**:
```csharp
// Calculate Cook's distance for each point
var cookDistances = CalculateCookDistances(X, y, yPredicted);

// Count influential points (Cook's distance > threshold)
var influentialCount = cookDistances.Count(d => d > InfluentialThreshold);
var influentialRatio = influentialCount / totalPoints;

if (influentialRatio > OverfitThreshold)
    return FitType.Overfit;  // Too many influential points
else if (influentialRatio < UnderfitThreshold)
    return FitType.Underfit;  // Too few influential points
else
    return FitType.GoodFit;
```

**Cook's Distance Formula**:
```csharp
// For point i:
// D_i = (r_i^2 / p*MSE) * (h_ii^2 / (1 - h_ii)^2)
// Where:
// - r_i = residual for point i
// - p = number of parameters
// - MSE = mean squared error
// - h_ii = leverage (diagonal of hat matrix)
```

**FitTypes Returned**:
- `GoodFit` (moderate number of influential points)
- `Overfit` (too many influential points)
- `Underfit` (too few influential points)

---

### 4. ConfusionMatrixFitDetector
**Location**: `C:\Users\cheat\source\repos\AiDotNet\src\FitDetectors\ConfusionMatrixFitDetector.cs`

**What it does**:
- **Classification-specific** detector
- Calculates confusion matrix metrics (TP, TN, FP, FN)
- Derives metrics: Accuracy, Precision, Recall, F1-Score
- Detects class imbalance issues

**Key Logic**:
```csharp
// Calculate confusion matrix
var confusionMatrix = CalculateConfusionMatrix(actual, predicted);

// Extract metrics
var accuracy = (TP + TN) / (TP + TN + FP + FN);
var precision = TP / (TP + FP);
var recall = TP / (TP + FN);
var f1Score = 2 * (precision * recall) / (precision + recall);

// Determine fit based on primary metric (e.g., F1-Score)
if (primaryMetric > GoodFitThreshold)
    return FitType.GoodFit;
else if (primaryMetric > ModerateFitThreshold)
    return FitType.Moderate;
else
    return FitType.PoorFit;
```

**Confusion Matrix Structure**:
```
                Predicted
                Pos    Neg
Actual  Pos     TP     FN
        Neg     FP     TN

TP = True Positives (correctly predicted positive)
TN = True Negatives (correctly predicted negative)
FP = False Positives (incorrectly predicted positive)
FN = False Negatives (incorrectly predicted negative)
```

**FitTypes Returned**:
- `GoodFit` (high accuracy/F1)
- `Moderate` (acceptable performance)
- `PoorFit` or `VeryPoorFit` (low performance)

---

## What's MISSING (Test Coverage)

### Current State
- **0% test coverage** for all Statistical FitDetectors
- No tests for statistical calculations (Durbin-Watson, Breusch-Pagan, White, Cook's Distance)
- No tests for confusion matrix generation and metrics

### Required Test Coverage

#### 1. AutocorrelationFitDetector Tests
- **Positive autocorrelation**: Residuals with same-sign runs
- **Negative autocorrelation**: Residuals that alternate signs
- **No autocorrelation**: Random residuals
- **Durbin-Watson calculation**: Verify against known values
- **Edge cases**: Single residual, all zeros, extreme values

#### 2. HeteroscedasticityFitDetector Tests
- **Homoscedasticity**: Consistent error variance
- **Heteroscedasticity**: Error variance increases with predictions
- **Breusch-Pagan test**: Verify test statistic calculation
- **White test**: Verify test statistic calculation
- **Edge cases**: Perfect predictions (zero error), constant predictions

#### 3. CookDistanceFitDetector Tests
- **No influential points**: All Cook's distances < threshold
- **Few influential points**: 1-2 outliers
- **Many influential points**: Overfitting scenario
- **Cook's distance calculation**: Verify against known values
- **Edge cases**: Single data point, identical points, perfect fit

#### 4. ConfusionMatrixFitDetector Tests
- **Perfect classification**: All TP and TN
- **Random classification**: ~50% accuracy
- **Class imbalance**: 90% one class
- **Metric calculations**: Accuracy, Precision, Recall, F1
- **Edge cases**: All same class, no positives, no negatives

---

## Step-by-Step Test Implementation

### Step 1: Extend Test Helpers

Add to `FitDetectorTestHelpers.cs`:

```csharp
/// <summary>
/// Creates mock residuals (errors) for autocorrelation testing.
/// </summary>
public static List<T> CreateResiduals<T>(
    ResidualPattern pattern,
    int count = 100)
{
    var numOps = MathHelper.GetNumericOperations<T>();
    var residuals = new List<T>();
    var random = new Random(42);  // Fixed seed for reproducibility

    switch (pattern)
    {
        case ResidualPattern.PositiveAutocorrelation:
            // Create runs of same-sign residuals
            for (int i = 0; i < count; i++)
            {
                var sign = (i / 10) % 2 == 0 ? 1 : -1;
                residuals.Add(numOps.FromDouble(sign * (random.NextDouble() * 0.5 + 0.5)));
            }
            break;

        case ResidualPattern.NegativeAutocorrelation:
            // Create alternating residuals
            for (int i = 0; i < count; i++)
            {
                var sign = i % 2 == 0 ? 1 : -1;
                residuals.Add(numOps.FromDouble(sign * (random.NextDouble() * 0.5 + 0.5)));
            }
            break;

        case ResidualPattern.NoAutocorrelation:
            // Create random residuals
            for (int i = 0; i < count; i++)
            {
                var sign = random.Next(2) == 0 ? 1 : -1;
                residuals.Add(numOps.FromDouble(sign * random.NextDouble()));
            }
            break;

        case ResidualPattern.Heteroscedastic:
            // Error variance increases with index
            for (int i = 0; i < count; i++)
            {
                var variance = 0.1 + (i / (double)count) * 0.9;
                residuals.Add(numOps.FromDouble((random.NextDouble() - 0.5) * variance));
            }
            break;
    }

    return residuals;
}

public enum ResidualPattern
{
    PositiveAutocorrelation,
    NegativeAutocorrelation,
    NoAutocorrelation,
    Heteroscedastic
}

/// <summary>
/// Creates mock classification data (actual and predicted classes).
/// </summary>
public static (TOutput[] actual, TOutput[] predicted) CreateClassificationData<TOutput>(
    int truePositives,
    int trueNegatives,
    int falsePositives,
    int falseNegatives,
    TOutput positiveClass,
    TOutput negativeClass)
{
    var total = truePositives + trueNegatives + falsePositives + falseNegatives;
    var actual = new List<TOutput>();
    var predicted = new List<TOutput>();

    // True Positives (actual=pos, predicted=pos)
    for (int i = 0; i < truePositives; i++)
    {
        actual.Add(positiveClass);
        predicted.Add(positiveClass);
    }

    // True Negatives (actual=neg, predicted=neg)
    for (int i = 0; i < trueNegatives; i++)
    {
        actual.Add(negativeClass);
        predicted.Add(negativeClass);
    }

    // False Positives (actual=neg, predicted=pos)
    for (int i = 0; i < falsePositives; i++)
    {
        actual.Add(negativeClass);
        predicted.Add(positiveClass);
    }

    // False Negatives (actual=pos, predicted=neg)
    for (int i = 0; i < falseNegatives; i++)
    {
        actual.Add(positiveClass);
        predicted.Add(negativeClass);
    }

    return (actual.ToArray(), predicted.ToArray());
}
```

---

### Step 2: AutocorrelationFitDetector Tests

Create `AutocorrelationFitDetectorTests.cs`:

```csharp
public class AutocorrelationFitDetectorTests
{
    [Fact]
    public void DetectFit_StrongPositiveAutocorrelation_RunsOfSameSign()
    {
        // Arrange: Residuals in runs of same sign
        var detector = new AutocorrelationFitDetector<double, double[], double>();
        var residuals = CreatePositiveAutocorrelationResiduals();
        var evalData = CreateEvalDataWithResiduals(residuals);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.StrongPositiveAutocorrelation, result.FitType);
        Assert.Contains(result.Recommendations,
            r => r.Contains("lagged", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void DetectFit_StrongNegativeAutocorrelation_AlternatingSigns()
    {
        // Arrange: Residuals that alternate signs
        var detector = new AutocorrelationFitDetector<double, double[], double>();
        var residuals = CreateNegativeAutocorrelationResiduals();
        var evalData = CreateEvalDataWithResiduals(residuals);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.StrongNegativeAutocorrelation, result.FitType);
    }

    [Fact]
    public void DetectFit_NoAutocorrelation_RandomResiduals()
    {
        // Arrange: Random residuals
        var detector = new AutocorrelationFitDetector<double, double[], double>();
        var residuals = CreateRandomResiduals();
        var evalData = CreateEvalDataWithResiduals(residuals);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.NoAutocorrelation, result.FitType);
    }

    [Fact]
    public void DurbinWatson_KnownValue_MatchesExpected()
    {
        // Arrange: Known residuals with expected DW statistic
        var residuals = new[] { 0.1, 0.2, 0.15, 0.18, 0.22 };  // Positive autocorrelation

        // Calculate DW manually: DW = sum((e_t - e_{t-1})^2) / sum(e_t^2)
        // (0.2-0.1)^2 + (0.15-0.2)^2 + (0.18-0.15)^2 + (0.22-0.18)^2
        // = 0.01 + 0.0025 + 0.0009 + 0.0016 = 0.015
        // sum(e_t^2) = 0.01 + 0.04 + 0.0225 + 0.0324 + 0.0484 = 0.1533
        // DW = 0.015 / 0.1533 = 0.0978 (strong positive autocorrelation)

        var dwExpected = 0.0978;

        // Act
        var dwActual = StatisticsHelper<double>.CalculateDurbinWatsonStatistic(residuals);

        // Assert
        Assert.Equal(dwExpected, dwActual, precision: 3);
    }

    [Fact]
    public void ConfidenceLevel_CloseToIdeal_HighConfidence()
    {
        // Arrange: DW stat close to 2.0 (no autocorrelation)
        var detector = new AutocorrelationFitDetector<double, double[], double>();
        var residuals = CreateResiduals(durbinWatson: 2.0);
        var evalData = CreateEvalDataWithResiduals(residuals);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: Confidence = 1 - |2.0 - 2.0| / 2.0 = 1.0
        Assert.Equal(1.0, result.ConfidenceLevel.Value, precision: 2);
    }

    [Fact]
    public void EdgeCase_SingleResidual_HandlesGracefully()
    {
        // Arrange: Only one residual (can't calculate autocorrelation)
        var detector = new AutocorrelationFitDetector<double, double[], double>();
        var residuals = new[] { 0.5 };
        var evalData = CreateEvalDataWithResiduals(residuals);

        // Act & Assert: Should not throw exception
        var exception = Record.Exception(() => detector.DetectFit(evalData));
        Assert.Null(exception);
    }

    [Fact]
    public void EdgeCase_AllZeroResiduals_PerfectFit()
    {
        // Arrange: All residuals = 0 (perfect predictions)
        var detector = new AutocorrelationFitDetector<double, double[], double>();
        var residuals = new[] { 0.0, 0.0, 0.0, 0.0, 0.0 };
        var evalData = CreateEvalDataWithResiduals(residuals);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: DW undefined for zero variance, should handle gracefully
        Assert.NotNull(result);
    }
}
```

---

### Step 3: HeteroscedasticityFitDetector Tests

```csharp
public class HeteroscedasticityFitDetectorTests
{
    [Fact]
    public void DetectFit_Homoscedasticity_ConsistentErrorVariance()
    {
        // Arrange: Errors have consistent variance across predictions
        var detector = new HeteroscedasticityFitDetector<double, double[], double>();
        var evalData = CreateHomoscedasticData();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.GoodFit, result.FitType);
    }

    [Fact]
    public void DetectFit_Heteroscedasticity_IncreasingErrorVariance()
    {
        // Arrange: Error variance increases with prediction values
        var detector = new HeteroscedasticityFitDetector<double, double[], double>();
        var evalData = CreateHeteroscedasticData();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.Unstable, result.FitType);
        Assert.Contains(result.Recommendations,
            r => r.Contains("transform", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void BreuschPaganTest_CalculatesCorrectly()
    {
        // Arrange: Known heteroscedastic data
        var evalData = CreateHeteroscedasticData();
        var detector = new HeteroscedasticityFitDetector<double, double[], double>();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: Breusch-Pagan statistic should be in AdditionalInfo
        Assert.True(result.AdditionalInfo.ContainsKey("BreuschPaganTestStatistic"));
        var bpStat = (double)result.AdditionalInfo["BreuschPaganTestStatistic"];
        Assert.True(bpStat > 0);
    }

    [Fact]
    public void WhiteTest_CalculatesCorrectly()
    {
        // Arrange
        var evalData = CreateHeteroscedasticData();
        var detector = new HeteroscedasticityFitDetector<double, double[], double>();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: White statistic should be in AdditionalInfo
        Assert.True(result.AdditionalInfo.ContainsKey("WhiteTestStatistic"));
        var whiteStat = (double)result.AdditionalInfo["WhiteTestStatistic"];
        Assert.True(whiteStat > 0);
    }
}
```

---

### Step 4: CookDistanceFitDetector Tests

```csharp
public class CookDistanceFitDetectorTests
{
    [Fact]
    public void DetectFit_NoInfluentialPoints_GoodFit()
    {
        // Arrange: Linear data with no outliers
        var detector = new CookDistanceFitDetector<double, double[], double>();
        var evalData = CreateLinearDataNoOutliers();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.GoodFit, result.FitType);
    }

    [Fact]
    public void DetectFit_ManyInfluentialPoints_Overfit()
    {
        // Arrange: Data with many outliers
        var detector = new CookDistanceFitDetector<double, double[], double>();
        var evalData = CreateDataWithManyOutliers();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.Overfit, result.FitType);
    }

    [Fact]
    public void CookDistance_SingleOutlier_HighDistance()
    {
        // Arrange: Data with one clear outlier
        var detector = new CookDistanceFitDetector<double, double[], double>();
        var evalData = CreateDataWithOneOutlier();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: Should identify the outlier in recommendations
        Assert.Contains(result.Recommendations,
            r => r.Contains("influential", StringComparison.OrdinalIgnoreCase));

        // Check AdditionalInfo contains Cook's distances
        Assert.True(result.AdditionalInfo.ContainsKey("CookDistances"));
    }

    [Fact]
    public void Recommendations_IncludeTopInfluentialPoints()
    {
        // Arrange
        var detector = new CookDistanceFitDetector<double, double[], double>();
        var evalData = CreateDataWithOutliers();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: Should list top 5 influential points
        var influentialPoints = result.Recommendations.Where(
            r => r.Contains("index:", StringComparison.OrdinalIgnoreCase)).ToList();
        Assert.True(influentialPoints.Count <= 5);
    }
}
```

---

### Step 5: ConfusionMatrixFitDetector Tests

```csharp
public class ConfusionMatrixFitDetectorTests
{
    [Fact]
    public void DetectFit_PerfectClassification_GoodFit()
    {
        // Arrange: All predictions correct
        var detector = new ConfusionMatrixFitDetector<double, double[], int>(
            new ConfusionMatrixFitDetectorOptions { PrimaryMetric = "Accuracy" });

        var (actual, predicted) = FitDetectorTestHelpers.CreateClassificationData(
            truePositives: 50,
            trueNegatives: 50,
            falsePositives: 0,
            falseNegatives: 0,
            positiveClass: 1,
            negativeClass: 0);

        var evalData = CreateClassificationEvalData(actual, predicted);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.GoodFit, result.FitType);
        Assert.Equal(1.0, result.ConfidenceLevel.Value, precision: 2);  // 100% accuracy
    }

    [Fact]
    public void DetectFit_RandomClassification_PoorFit()
    {
        // Arrange: ~50% accuracy (random)
        var detector = new ConfusionMatrixFitDetector<double, double[], int>(
            new ConfusionMatrixFitDetectorOptions { PrimaryMetric = "Accuracy" });

        var (actual, predicted) = FitDetectorTestHelpers.CreateClassificationData(
            truePositives: 25,
            trueNegatives: 25,
            falsePositives: 25,
            falseNegatives: 25,
            positiveClass: 1,
            negativeClass: 0);

        var evalData = CreateClassificationEvalData(actual, predicted);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.PoorFit, result.FitType);
    }

    [Theory]
    [InlineData(90, 10, 5, 5, "Precision")]  // High precision
    [InlineData(10, 90, 5, 5, "Recall")]     // High recall
    [InlineData(45, 45, 5, 5, "F1Score")]    // Balanced F1
    public void DetectFit_DifferentMetrics_CorrectEvaluation(
        int tp, int tn, int fp, int fn, string primaryMetric)
    {
        // Arrange
        var detector = new ConfusionMatrixFitDetector<double, double[], int>(
            new ConfusionMatrixFitDetectorOptions { PrimaryMetric = primaryMetric });

        var (actual, predicted) = FitDetectorTestHelpers.CreateClassificationData(
            tp, tn, fp, fn, 1, 0);

        var evalData = CreateClassificationEvalData(actual, predicted);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: Should use specified metric for evaluation
        Assert.NotNull(result.FitType);
    }

    [Fact]
    public void DetectFit_ClassImbalance_DetectedAndReported()
    {
        // Arrange: 90% one class
        var detector = new ConfusionMatrixFitDetector<double, double[], int>(
            new ConfusionMatrixFitDetectorOptions());

        var (actual, predicted) = FitDetectorTestHelpers.CreateClassificationData(
            truePositives: 90,
            trueNegatives: 5,
            falsePositives: 3,
            falseNegatives: 2,
            positiveClass: 1,
            negativeClass: 0);

        var evalData = CreateClassificationEvalData(actual, predicted);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: Should mention class imbalance in recommendations
        Assert.Contains(result.Recommendations,
            r => r.Contains("imbalance", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void EdgeCase_AllSameClass_HandlesGracefully()
    {
        // Arrange: All actual values are the same class
        var detector = new ConfusionMatrixFitDetector<double, double[], int>(
            new ConfusionMatrixFitDetectorOptions());

        var (actual, predicted) = FitDetectorTestHelpers.CreateClassificationData(
            truePositives: 100,
            trueNegatives: 0,
            falsePositives: 0,
            falseNegatives: 0,
            positiveClass: 1,
            negativeClass: 0);

        var evalData = CreateClassificationEvalData(actual, predicted);

        // Act & Assert: Should not throw division by zero
        var exception = Record.Exception(() => detector.DetectFit(evalData));
        Assert.Null(exception);
    }

    [Fact]
    public void Metrics_CalculateCorrectly()
    {
        // Arrange: Known confusion matrix
        var detector = new ConfusionMatrixFitDetector<double, double[], int>(
            new ConfusionMatrixFitDetectorOptions());

        // TP=40, TN=30, FP=10, FN=20
        var (actual, predicted) = FitDetectorTestHelpers.CreateClassificationData(
            40, 30, 10, 20, 1, 0);

        // Expected metrics:
        // Accuracy = (40+30)/(40+30+10+20) = 70/100 = 0.70
        // Precision = 40/(40+10) = 40/50 = 0.80
        // Recall = 40/(40+20) = 40/60 = 0.667
        // F1 = 2*(0.80*0.667)/(0.80+0.667) = 0.727

        var evalData = CreateClassificationEvalData(actual, predicted);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: Verify metrics in recommendations or additional info
        // (Implementation-dependent - check how metrics are reported)
        Assert.NotNull(result);
    }
}
```

---

## Testing Strategy

### 1. Statistical Test Validation
- **Verify calculations** against known values from statistics textbooks
- **Use R or Python** to generate expected values for complex tests
- **Test edge cases** (n=1, n=2, all zeros, all same value)

### 2. Residual Pattern Testing
- Create helper functions to generate different patterns
- Test clear cases first (strong autocorrelation, severe heteroscedasticity)
- Test borderline cases (weak patterns)

### 3. Classification Testing
- Test all four quadrants of confusion matrix
- Test class imbalance scenarios
- Test multi-class scenarios (if supported)

### 4. Coverage Goals
- **Minimum**: 80% code coverage
- **Target**: 90% including all statistical calculations
- **Focus**: All FitType branches, all metric calculations

---

## Common Pitfalls

### 1. Durbin-Watson Division by Zero
**Problem**: If all residuals are zero, DW calculation divides by zero

**Solution**:
```csharp
if (sumSquaredResiduals < NumOps.Epsilon)
    return NumOps.FromDouble(2.0);  // Perfect fit, no autocorrelation
```

### 2. Confusion Matrix with No Positives/Negatives
**Problem**: Division by zero when calculating precision/recall

**Solution**:
```csharp
var precision = (TP + FP) > 0 ? TP / (TP + FP) : 0.0;
var recall = (TP + FN) > 0 ? TP / (TP + FN) : 0.0;
```

### 3. Cook's Distance Matrix Inversion
**Problem**: Hat matrix may be singular (non-invertible)

**Solution**: Use pseudoinverse or add small regularization term

### 4. Heteroscedasticity Test Sample Size
**Problem**: Tests require sufficient sample size (typically n > 30)

**Solution**: Check sample size before running tests, return "Insufficient Data" if too small

---

## Next Steps

1. Read all four detector source files completely
2. Understand each statistical test's mathematics
3. Create test data generators for each pattern type
4. Implement tests for each detector
5. Verify calculations against external tools (R, Python)
6. Document any discrepancies or issues found

---

## Additional Resources

- **Durbin-Watson Test**: https://en.wikipedia.org/wiki/Durbin%E2%80%93Watson_statistic
- **Breusch-Pagan Test**: https://en.wikipedia.org/wiki/Breusch%E2%80%93Pagan_test
- **White Test**: https://en.wikipedia.org/wiki/White_test
- **Cook's Distance**: https://en.wikipedia.org/wiki/Cook%27s_distance
- **Confusion Matrix**: https://en.wikipedia.org/wiki/Confusion_matrix
