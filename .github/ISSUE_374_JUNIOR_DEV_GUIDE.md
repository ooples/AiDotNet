# Junior Developer Implementation Guide: Model-Based Fit Detectors (Issue #374)

## Table of Contents
1. [For Beginners: What Are Model-Based Fit Detectors?](#for-beginners-what-are-model-based-fit-detectors)
2. [What EXISTS in src/FitDetectors/](#what-exists-in-srcfitdetectors)
3. [What's MISSING (Test Coverage)](#whats-missing-test-coverage)
4. [Step-by-Step Test Implementation](#step-by-step-test-implementation)
5. [Testing Strategy](#testing-strategy)
6. [Common Pitfalls](#common-pitfalls)

---

## For Beginners: What Are Model-Based Fit Detectors?

### Advanced Model Analysis
**Model-based fit detectors** use sophisticated machine learning techniques to analyze fit:

1. **Gaussian Process**: Analyzes prediction uncertainty
   - Provides confidence intervals for predictions
   - Detects when model is uncertain vs. confidently wrong

2. **Gradient Boosting**: Uses ensemble learning insights
   - Analyzes how many boosting iterations are needed
   - Detects when additional models stop improving performance

3. **Ensemble**: Combines multiple detectors
   - Voting or averaging across different detection methods
   - More robust than single detector

4. **Feature Importance**: Analyzes which features matter
   - Identifies irrelevant features (sign of overfitting)
   - Detects if model depends too heavily on one feature

5. **Calibrated Probability**: Tests prediction confidence quality
   - For classification: Are 90% confidence predictions actually right 90% of the time?
   - Detects overconfident or underconfident models

---

## What EXISTS in src/FitDetectors/

### 1. GaussianProcessFitDetector
**Location**: `C:\Users\cheat\source\repos\AiDotNet\src\FitDetectors\GaussianProcessFitDetector.cs`

**What it does**:
- Performs Gaussian Process regression on evaluation data
- Calculates mean predictions AND variance (uncertainty) predictions
- Uses RBF (Radial Basis Function) kernel
- Optimizes hyperparameters (length scale, noise variance) via grid search

**Key Logic**:
```csharp
// Perform GP regression
var (meanPrediction, variancePrediction) = PerformGaussianProcessRegression(evalData);

var averageUncertainty = variancePrediction.Average();
var rmse = CalculateRMSE(actual, meanPrediction);

// Good fit: Low RMSE + Low uncertainty
if (rmse < GoodFitThreshold && averageUncertainty < LowUncertaintyThreshold)
    return FitType.GoodFit;

// Overfit: High RMSE + Low uncertainty (confidently wrong!)
if (rmse > OverfitThreshold && averageUncertainty < LowUncertaintyThreshold)
    return FitType.Overfit;

// Underfit: High RMSE + High uncertainty
if (rmse > UnderfitThreshold && averageUncertainty > HighUncertaintyThreshold)
    return FitType.Underfit;

return FitType.Unstable;
```

**Gaussian Process Steps**:
1. Optimize hyperparameters (length scale λ, noise variance σ²)
2. Calculate kernel matrix K using RBF kernel
3. Add noise to diagonal for stability
4. Perform Cholesky decomposition: K = L L^T
5. Solve for α: K α = y → α = K^(-1) y
6. Calculate mean predictions: μ* = K* α
7. Calculate variance predictions: σ*² = k** - v^T v

**RBF Kernel**:
```csharp
k(x1, x2) = exp(-||x1 - x2||² / (2 λ²))
```

**FitTypes Returned**:
- `GoodFit` (low error, low uncertainty)
- `Overfit` (high error, low uncertainty - confidently wrong)
- `Underfit` (high error, high uncertainty)
- `Unstable` (mixed signals)

---

### 2. GradientBoostingFitDetector
**Location**: `C:\Users\cheat\source\repos\AiDotNet\src\FitDetectors\GradientBoostingFitDetector.cs`

**What it does**:
- Analyzes learning curves from boosting iterations
- Tracks training and validation error over iterations
- Detects when validation error stops improving (plateau)
- Identifies early stopping point

**Key Concepts**:
```csharp
// Train gradient boosting, tracking error at each iteration
var trainingErrors = new List<T>();
var validationErrors = new List<T>();

for (int iter = 0; iter < maxIterations; iter++)
{
    // Add weak learner
    boosting.AddLearner();

    trainingErrors.Add(CalculateError(trainingSet));
    validationErrors.Add(CalculateError(validationSet));
}

// Analyze learning curves
var trainingErrorDecrease = trainingErrors[0] - trainingErrors[last];
var validationErrorDecrease = validationErrors[0] - validationErrors[last];

// Good fit: Both errors decrease together
if (trainingErrorDecrease > threshold && validationErrorDecrease > threshold)
    return FitType.GoodFit;

// Overfit: Training decreases, validation plateaus or increases
if (trainingErrorDecrease > threshold && validationErrorDecrease < smallThreshold)
    return FitType.Overfit;

// Underfit: Neither error decreases much
if (trainingErrorDecrease < smallThreshold && validationErrorDecrease < smallThreshold)
    return FitType.Underfit;
```

**FitTypes Returned**:
- `GoodFit` (both errors decrease)
- `Overfit` (training improves, validation doesn't)
- `Underfit` (neither improves)

---

### 3. EnsembleFitDetector
**Location**: `C:\Users\cheat\source\repos\AiDotNet\src\FitDetectors\EnsembleFitDetector.cs`

**What it does**:
- Combines multiple fit detectors
- Uses voting or averaging to determine final FitType
- More robust than any single detector

**Key Logic**:
```csharp
// Create ensemble of detectors
var detectors = new IFitDetector<T, TInput, TOutput>[]
{
    new DefaultFitDetector<T, TInput, TOutput>(),
    new HoldoutValidationFitDetector<T, TInput, TOutput>(),
    new BootstrapFitDetector<T, TInput, TOutput>(),
    new GaussianProcessFitDetector<T, TInput, TOutput>()
};

// Get results from all detectors
var results = detectors.Select(d => d.DetectFit(evalData)).ToList();

// Voting strategy
var fitTypeCounts = results.GroupBy(r => r.FitType)
                          .OrderByDescending(g => g.Count())
                          .ToList();

var majorityFitType = fitTypeCounts.First().Key;
var consensusStrength = fitTypeCounts.First().Count() / (double)detectors.Length;

// Confidence based on consensus
var confidence = consensusStrength;  // e.g., 0.75 if 3/4 agree
```

**Ensemble Strategies**:
- **Majority Voting**: Most common FitType
- **Weighted Voting**: Weight by detector confidence
- **Unanimous**: Only return FitType if all agree

**FitTypes Returned**: Any FitType (depends on ensemble)

---

### 4. FeatureImportanceFitDetector
**Location**: `C:\Users\cheat\source\repos\AiDotNet\src\FitDetectors\FeatureImportanceFitDetector.cs`

**What it does**:
- Calculates importance of each feature
- Uses permutation importance (shuffle feature, measure error increase)
- Detects if model overfits to specific features

**Key Logic**:
```csharp
// Calculate baseline error
var baselineError = CalculateError(model, X, y);

// For each feature
var importances = new List<double>();
foreach (int featureIndex in Enumerable.Range(0, numFeatures))
{
    // Shuffle this feature
    var XShuffled = ShuffleFeature(X, featureIndex);

    // Calculate new error
    var shuffledError = CalculateError(model, XShuffled, y);

    // Importance = how much error increased
    var importance = shuffledError - baselineError;
    importances.Add(importance);
}

// Analyze importance distribution
var maxImportance = importances.Max();
var avgImportance = importances.Average();
var importanceRatio = maxImportance / avgImportance;

// Overfit: Model depends heavily on few features
if (importanceRatio > OverfitThreshold)
    return FitType.Overfit;

// Underfit: All features have low importance
if (maxImportance < UnderfitThreshold)
    return FitType.Underfit;

// Good fit: Balanced importance across features
if (importanceRatio < GoodFitThreshold && maxImportance > MinThreshold)
    return FitType.GoodFit;
```

**FitTypes Returned**:
- `GoodFit` (balanced feature importance)
- `Overfit` (too dependent on few features)
- `Underfit` (no features are important)
- `HighVariance` (importance unstable)

---

### 5. CalibratedProbabilityFitDetector
**Location**: `C:\Users\cheat\source\repos\AiDotNet\src\FitDetectors\CalibratedProbabilityFitDetector.cs`

**What it does**:
- **Classification models only**
- Tests if predicted probabilities match actual outcomes
- Uses calibration curves (reliability diagrams)
- Calculates Expected Calibration Error (ECE)

**Key Logic**:
```csharp
// Bin predictions by confidence level
var bins = CreateBins(numBins: 10);  // [0-0.1, 0.1-0.2, ..., 0.9-1.0]

foreach (var (predicted, actual) in predictions.Zip(actuals))
{
    var binIndex = (int)(predicted * numBins);
    bins[binIndex].Add((predicted, actual));
}

// Calculate calibration error for each bin
var calibrationErrors = new List<double>();
foreach (var bin in bins)
{
    var avgPredicted = bin.Average(p => p.predicted);  // Average confidence
    var avgActual = bin.Average(p => p.actual);        // Actual accuracy

    var error = |avgPredicted - avgActual|;
    calibrationErrors.Add(error);
}

// Expected Calibration Error
var ece = calibrationErrors.Average();

// Well-calibrated: ECE < threshold
if (ece < WellCalibratedThreshold)
    return FitType.GoodFit;

// Overconfident: Predictions higher than actual accuracy
if (avgPredicted > avgActual + threshold)
    return FitType.Overfit;

// Underconfident: Predictions lower than actual accuracy
if (avgPredicted < avgActual - threshold)
    return FitType.Underfit;
```

**Calibration Curve**:
- X-axis: Predicted probability
- Y-axis: Actual frequency of positive class
- Perfect calibration: y = x (diagonal line)
- Overconfident: Curve below diagonal
- Underconfident: Curve above diagonal

**FitTypes Returned**:
- `GoodFit` (well-calibrated)
- `Overfit` (overconfident)
- `Underfit` (underconfident)
- `Unstable` (poor calibration)

---

## What's MISSING (Test Coverage)

### Current State
- **0% test coverage** for all Model-Based FitDetectors
- No tests for GP kernel calculations, Cholesky decomposition
- No tests for feature permutation importance
- No tests for calibration curves

### Required Test Coverage

#### 1. GaussianProcessFitDetector Tests
- **RBF kernel calculation**: Verify against known values
- **Hyperparameter optimization**: Grid search finds optimal values
- **Cholesky decomposition**: Handles singular matrices
- **Mean/variance predictions**: Match expected values
- **FitType logic**: Low error + low uncertainty = GoodFit, etc.

#### 2. GradientBoostingFitDetector Tests
- **Learning curves**: Training and validation errors over iterations
- **Early stopping detection**: Identifies when to stop
- **Overfit detection**: Validation error increases while training decreases
- **Underfit detection**: Neither error decreases

#### 3. EnsembleFitDetector Tests
- **Voting mechanism**: Majority vote works correctly
- **Consensus strength**: Confidence reflects agreement level
- **All detectors agree**: 100% confidence
- **Split decision**: Lower confidence
- **Unanimous requirement**: Works when configured

#### 4. FeatureImportanceFitDetector Tests
- **Permutation importance**: Shuffling reduces accuracy
- **Important vs. irrelevant features**: Detected correctly
- **Overfit to one feature**: High importance ratio
- **Balanced importance**: GoodFit

#### 5. CalibratedProbabilityFitDetector Tests
- **Perfect calibration**: y = x diagonal
- **Overconfident model**: Curve below diagonal
- **Underconfident model**: Curve above diagonal
- **ECE calculation**: Matches expected value
- **Binning strategy**: Handles edge cases

---

## Step-by-Step Test Implementation

### Step 1: GaussianProcessFitDetector Tests

```csharp
public class GaussianProcessFitDetectorTests
{
    [Fact]
    public void RBFKernel_IdenticalPoints_Returns1()
    {
        // Arrange
        var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var lengthScale = 1.0;

        // Act
        var kernel = CalculateRBFKernel(x, x, lengthScale);

        // Assert: k(x, x) = exp(0) = 1
        Assert.Equal(1.0, kernel, precision: 6);
    }

    [Fact]
    public void RBFKernel_DistantPoints_ReturnsNearZero()
    {
        // Arrange
        var x1 = new Vector<double>(new[] { 0.0, 0.0 });
        var x2 = new Vector<double>(new[] { 10.0, 10.0 });
        var lengthScale = 1.0;

        // Act
        var kernel = CalculateRBFKernel(x1, x2, lengthScale);

        // Assert: exp(-||x1-x2||^2 / (2*λ^2)) ≈ 0 for large distance
        Assert.True(kernel < 0.01);
    }

    [Fact]
    public void DetectFit_LowErrorLowUncertainty_GoodFit()
    {
        // Arrange: Data that GP can fit well
        var detector = new GaussianProcessFitDetector<double, double[], double>();
        var evalData = CreateLinearDataWithNoise(noise: 0.01);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.GoodFit, result.FitType);
    }

    [Fact]
    public void DetectFit_HighErrorLowUncertainty_Overfit()
    {
        // Arrange: GP is confident but wrong
        var detector = new GaussianProcessFitDetector<double, double[], double>();
        var evalData = CreateMismatchedData();  // Training and test from different distributions

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.Overfit, result.FitType);
    }

    [Fact]
    public void DetectFit_HighErrorHighUncertainty_Underfit()
    {
        // Arrange: Complex nonlinear data that GP struggles with
        var detector = new GaussianProcessFitDetector<double, double[], double>();
        var evalData = CreateComplexNonlinearData();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.Underfit, result.FitType);
    }

    [Fact]
    public void HyperparameterOptimization_FindsReasonableValues()
    {
        // Arrange
        var detector = new GaussianProcessFitDetector<double, double[], double>();
        var evalData = CreateLinearDataWithNoise(noise: 0.1);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: Optimized hyperparameters should be reasonable
        // (Check via logging or reflection if exposed)
        Assert.NotNull(result);
    }

    [Fact]
    public void CholeskyDecomposition_SingularMatrix_HandlesGracefully()
    {
        // Arrange: Data that creates singular kernel matrix
        var detector = new GaussianProcessFitDetector<double, double[], double>();
        var evalData = CreateIdenticalPoints();  // All points the same

        // Act & Assert: Should not throw exception
        var exception = Record.Exception(() => detector.DetectFit(evalData));
        Assert.Null(exception);
    }

    [Fact]
    public void Recommendations_IncludeKernelSuggestions()
    {
        // Arrange
        var detector = new GaussianProcessFitDetector<double, double[], double>();
        var evalData = CreateLinearDataWithNoise();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: Should mention kernel functions
        var hasKernelMention = result.Recommendations.Any(
            r => r.Contains("kernel", StringComparison.OrdinalIgnoreCase));
        Assert.True(hasKernelMention);
    }
}
```

---

### Step 2: FeatureImportanceFitDetector Tests

```csharp
public class FeatureImportanceFitDetectorTests
{
    [Fact]
    public void DetectFit_OneImportantFeature_Overfit()
    {
        // Arrange: Model that only uses feature 0
        var detector = new FeatureImportanceFitDetector<double, double[], double>();
        var evalData = CreateDataWithOneImportantFeature();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: High importance ratio suggests overfitting
        Assert.Equal(FitType.Overfit, result.FitType);
    }

    [Fact]
    public void DetectFit_BalancedImportance_GoodFit()
    {
        // Arrange: All features contribute
        var detector = new FeatureImportanceFitDetector<double, double[], double>();
        var evalData = CreateDataWithBalancedFeatures();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.GoodFit, result.FitType);
    }

    [Fact]
    public void DetectFit_NoImportantFeatures_Underfit()
    {
        // Arrange: All features have low importance
        var detector = new FeatureImportanceFitDetector<double, double[], double>();
        var evalData = CreateDataWithNoSignal();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.Underfit, result.FitType);
    }

    [Fact]
    public void PermutationImportance_ImportantFeature_LargeDecrease()
    {
        // Arrange: Feature 0 is highly predictive
        var model = TrainModelOnFeature0();
        var X = CreateDataMatrix();
        var y = CreateTargets();

        // Act: Shuffle feature 0
        var baselineError = CalculateError(model, X, y);
        var XShuffled = ShuffleFeature(X, featureIndex: 0);
        var shuffledError = CalculateError(model, XShuffled, y);

        // Assert: Error should increase significantly
        Assert.True(shuffledError > baselineError * 1.5);  // At least 50% worse
    }

    [Fact]
    public void PermutationImportance_IrrelevantFeature_NoChange()
    {
        // Arrange: Feature 1 is not used by model
        var model = TrainModelOnFeature0();
        var X = CreateDataMatrix();
        var y = CreateTargets();

        // Act: Shuffle feature 1 (irrelevant)
        var baselineError = CalculateError(model, X, y);
        var XShuffled = ShuffleFeature(X, featureIndex: 1);
        var shuffledError = CalculateError(model, XShuffled, y);

        // Assert: Error should not change much
        Assert.InRange(shuffledError, baselineError * 0.95, baselineError * 1.05);
    }
}
```

---

### Step 3: EnsembleFitDetector Tests

```csharp
public class EnsembleFitDetectorTests
{
    [Fact]
    public void DetectFit_AllDetectorsAgree_HighConfidence()
    {
        // Arrange: Clear overfit scenario - all detectors should agree
        var detector = new EnsembleFitDetector<double, double[], double>();
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(
            trainingR2: 0.98,
            validationR2: 0.55,
            testR2: 0.52);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.Overfit, result.FitType);
        Assert.True(result.ConfidenceLevel >= 0.8);  // High consensus
    }

    [Fact]
    public void DetectFit_DetectorsDisagree_LowerConfidence()
    {
        // Arrange: Borderline case - detectors may disagree
        var detector = new EnsembleFitDetector<double, double[], double>();
        var evalData = FitDetectorTestHelpers.CreateMockEvaluationData(
            trainingR2: 0.75,
            validationR2: 0.72,
            testR2: 0.68);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: Confidence reflects disagreement
        Assert.NotNull(result.ConfidenceLevel);
        // (May be lower than unanimous case)
    }

    [Fact]
    public void Voting_MajorityWins()
    {
        // Arrange: 3 detectors say Overfit, 1 says GoodFit
        var detector = new EnsembleFitDetector<double, double[], double>(
            new EnsembleFitDetectorOptions { VotingStrategy = VotingStrategy.Majority });

        var evalData = CreateBorderlineOverfitData();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: Should pick majority (Overfit)
        Assert.Equal(FitType.Overfit, result.FitType);
    }

    [Fact]
    public void Voting_Unanimous_RequiresAllAgree()
    {
        // Arrange
        var detector = new EnsembleFitDetector<double, double[], double>(
            new EnsembleFitDetectorOptions { VotingStrategy = VotingStrategy.Unanimous });

        var evalData = CreateBorderlineData();  // Detectors may disagree

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: If not unanimous, should return Unstable or lower confidence
        // (Implementation-dependent)
        Assert.NotNull(result);
    }
}
```

---

### Step 4: CalibratedProbabilityFitDetector Tests

```csharp
public class CalibratedProbabilityFitDetectorTests
{
    [Fact]
    public void DetectFit_PerfectCalibration_GoodFit()
    {
        // Arrange: Predicted probabilities match actual frequencies
        var detector = new CalibratedProbabilityFitDetector<double, double[], int>();

        // 90% confident predictions are right 90% of the time
        var (predicted, actual) = CreatePerfectlyCalib ratedPredictions();
        var evalData = CreateClassificationEvalData(predicted, actual);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.GoodFit, result.FitType);
    }

    [Fact]
    public void DetectFit_Overconfident_Overfit()
    {
        // Arrange: Model predicts 90% confidence but only right 60% of time
        var detector = new CalibratedProbabilityFitDetector<double, double[], int>();

        var predicted = Enumerable.Repeat(0.9, 100).ToArray();
        var actual = GenerateActuals(trueRate: 0.6, count: 100);
        var evalData = CreateClassificationEvalData(predicted, actual);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.Overfit, result.FitType);
    }

    [Fact]
    public void DetectFit_Underconfident_Underfit()
    {
        // Arrange: Model predicts 50% confidence but actually right 85% of time
        var detector = new CalibratedProbabilityFitDetector<double, double[], int>();

        var predicted = Enumerable.Repeat(0.5, 100).ToArray();
        var actual = GenerateActuals(trueRate: 0.85, count: 100);
        var evalData = CreateClassificationEvalData(predicted, actual);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.Underfit, result.FitType);
    }

    [Fact]
    public void ECE_Calculation_MatchesExpected()
    {
        // Arrange: Known calibration curve
        var predicted = new[] { 0.1, 0.2, 0.3, 0.7, 0.8, 0.9 };
        var actual = new[] { 0, 0, 0, 1, 1, 1 };  // Binary

        // Expected: Bin [0-0.5]: avg_pred=0.2, avg_actual=0.0, error=0.2
        //           Bin [0.5-1.0]: avg_pred=0.8, avg_actual=1.0, error=0.2
        // ECE = (0.2 + 0.2) / 2 = 0.2

        // Act
        var ece = CalculateECE(predicted, actual, numBins: 2);

        // Assert
        Assert.Equal(0.2, ece, precision: 2);
    }

    [Fact]
    public void CalibrationCurve_DiagonalLine_PerfectCalibration()
    {
        // Arrange
        var predicted = Enumerable.Range(0, 10).Select(i => i / 10.0).ToArray();
        var actual = CreateActualsMatchingPredictions(predicted);

        // Act
        var (binCenters, binAccuracies) = CreateCalibrationCurve(predicted, actual);

        // Assert: Should form diagonal line (y ≈ x)
        for (int i = 0; i < binCenters.Length; i++)
        {
            Assert.Equal(binCenters[i], binAccuracies[i], precision: 1);
        }
    }
}
```

---

## Testing Strategy

### 1. Mathematical Correctness
- Verify kernel calculations against reference implementations
- Test matrix operations (Cholesky, inverse) on known matrices
- Compare calibration metrics to scikit-learn output

### 2. Model Training for Tests
- Use simple synthetic datasets (linear, quadratic)
- Train lightweight models for feature importance tests
- Mock GP predictions if training is too slow

### 3. Performance Testing
- GP with 100 points should complete in <5 seconds
- Ensemble of 4 detectors should complete in <10 seconds
- Feature importance with 10 features should complete in <3 seconds

### 4. Coverage Goals
- **Minimum**: 80% code coverage
- **Target**: 90% including all mathematical operations
- **Focus**: All FitType branches, numerical stability

---

## Common Pitfalls

### 1. Cholesky Decomposition Failure
**Problem**: Non-positive-definite matrix

**Solution**: Add small value to diagonal (jitter)
```csharp
for (int i = 0; i < K.Rows; i++)
    K[i, i] += 1e-6;  // Numerical stability
```

### 2. GP Hyperparameter Optimization Slowness
**Problem**: Grid search over many parameters is slow

**Solution**:
- Use coarse grid for tests (5x5 instead of 100x100)
- Cache GP results for repeated tests
- Use smaller datasets (n=50 instead of n=1000)

### 3. Feature Permutation Randomness
**Problem**: Tests fail intermittently due to random shuffling

**Solution**: Use fixed random seed
```csharp
var random = new Random(seed: 42);
```

### 4. Calibration with Imbalanced Classes
**Problem**: Bins have very few samples

**Solution**: Use adaptive binning or minimum bin size

### 5. Ensemble Voting Ties
**Problem**: 2 detectors say Overfit, 2 say Underfit

**Solution**:
- Implement tie-breaking rules
- Use weighted voting by confidence
- Return "Unstable" for ties

---

## Next Steps

1. Review each detector's source code completely
2. Understand mathematical foundations (GP regression, calibration)
3. Create test data generators for each scenario
4. Implement comprehensive tests
5. Benchmark performance (ensure tests complete in reasonable time)
6. Document any numerical stability issues found

---

## Additional Resources

- **Gaussian Process**: http://gaussianprocess.org/
- **Feature Importance**: https://christophm.github.io/interpretable-ml-book/
- **Model Calibration**: https://scikit-learn.org/stable/modules/calibration.html
- **RBF Kernel**: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
- **Expected Calibration Error**: https://arxiv.org/abs/1706.04599
