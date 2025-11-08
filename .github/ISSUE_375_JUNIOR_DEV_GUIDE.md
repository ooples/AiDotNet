# Junior Developer Implementation Guide: Advanced Fit Detectors (Issue #375)

## Table of Contents
1. [For Beginners: What Are Advanced Fit Detectors?](#for-beginners-what-are-advanced-fit-detectors)
2. [What EXISTS in src/FitDetectors/](#what-exists-in-srcfitdetectors)
3. [What's MISSING (Test Coverage)](#whats-missing-test-coverage)
4. [Step-by-Step Test Implementation](#step-by-step-test-implementation)
5. [Testing Strategy](#testing-strategy)
6. [Common Pitfalls](#common-pitfalls)

---

## For Beginners: What Are Advanced Fit Detectors?

### Cutting-Edge Techniques
**Advanced fit detectors** represent the most sophisticated approaches to fit detection:

1. **Bayesian**: Uses Bayesian statistics for model comparison
   - DIC (Deviance Information Criterion)
   - WAIC (Widely Applicable Information Criterion)
   - LOO (Leave-One-Out cross-validation)
   - Accounts for model complexity AND uncertainty

2. **Information Criteria**: Statistical measures that balance fit and complexity
   - AIC (Akaike Information Criterion)
   - BIC (Bayesian Information Criterion)
   - Lower values = better models

3. **Adaptive**: Dynamically adjusts thresholds based on data
   - Learns from model performance patterns
   - Adapts to different data distributions
   - More robust than fixed thresholds

4. **Hybrid**: Combines multiple detection approaches intelligently
   - Uses different detectors for different scenarios
   - Switches strategy based on data characteristics
   - Most flexible and powerful approach

---

## What EXISTS in src/FitDetectors/

### 1. BayesianFitDetector
**Location**: `C:\Users\cheat\source\repos\AiDotNet\src\FitDetectors\BayesianFitDetector.cs`

**What it does**:
- Calculates Bayesian model comparison metrics
- Uses DIC, WAIC, and LOO to assess fit
- Considers both data fit and model complexity
- Provides probabilistic interpretation of fit quality

**Key Metrics**:

**DIC (Deviance Information Criterion)**:
```csharp
// DIC = D(θ̄) + pD
// Where:
// D(θ̄) = -2 * log(p(y|θ̄))  (deviance at posterior mean)
// pD = D̄ - D(θ̄)              (effective number of parameters)

var posteriorMean = CalculatePosteriorMean(samples);
var deviance = -2 * LogLikelihood(data, posteriorMean);
var meanDeviance = samples.Average(s => -2 * LogLikelihood(data, s));
var pD = meanDeviance - deviance;
var dic = deviance + pD;
```

**WAIC (Widely Applicable Information Criterion)**:
```csharp
// WAIC = -2 * (lppd - pWAIC)
// Where:
// lppd = log pointwise predictive density
// pWAIC = variance of log likelihoods

var logLikelihoods = CalculateLogLikelihoods(data, samples);
var lppd = logLikelihoods.Select(ll => Log(Mean(Exp(ll)))).Sum();
var pWAIC = logLikelihoods.Select(ll => Variance(ll)).Sum();
var waic = -2 * (lppd - pWAIC);
```

**LOO (Leave-One-Out cross-validation)**:
```csharp
// LOO ≈ -2 * Σ log(p(y_i | y_-i))
// Approximated using Pareto-smoothed importance sampling

var looValues = new List<double>();
foreach (int i in Enumerable.Range(0, n))
{
    var yWithoutI = RemovePoint(y, i);
    var likelihood = CalculateLikelihood(y[i], yWithoutI, samples);
    looValues.Add(Log(likelihood));
}
var loo = -2 * looValues.Sum();
```

**Key Logic**:
```csharp
var dic = CalculateDIC(evalData);
var waic = CalculateWAIC(evalData);
var loo = CalculateLOO(evalData);

// All three metrics agree on good fit
if (dic < GoodFitThreshold && waic < GoodFitThreshold && loo < GoodFitThreshold)
    return FitType.GoodFit;

// High metrics suggest overfitting or poor fit
if (dic > OverfitThreshold || waic > OverfitThreshold || loo > OverfitThreshold)
    return FitType.Overfit;

// Metrics in middle range suggest underfitting
if (dic > UnderfitThreshold && waic > UnderfitThreshold && loo > UnderfitThreshold)
    return FitType.Underfit;

return FitType.Unstable;
```

**FitTypes Returned**:
- `GoodFit` (low DIC/WAIC/LOO)
- `Overfit` (very high metrics)
- `Underfit` (high metrics)
- `Unstable` (inconsistent metrics)

---

### 2. InformationCriteriaFitDetector
**Location**: `C:\Users\cheat\source\repos\AiDotNet\src\FitDetectors\InformationCriteriaFitDetector.cs`

**What it does**:
- Calculates AIC and BIC
- Compares model against baseline (null model)
- Determines if model complexity is justified

**Key Metrics**:

**AIC (Akaike Information Criterion)**:
```csharp
// AIC = 2k - 2ln(L)
// Where:
// k = number of parameters
// L = maximum likelihood

var k = CountParameters(model);
var logLikelihood = CalculateLogLikelihood(data, model);
var aic = 2 * k - 2 * logLikelihood;
```

**BIC (Bayesian Information Criterion)**:
```csharp
// BIC = k*ln(n) - 2ln(L)
// Where:
// k = number of parameters
// n = number of data points
// L = maximum likelihood

var k = CountParameters(model);
var n = data.Count;
var logLikelihood = CalculateLogLikelihood(data, model);
var bic = k * Log(n) - 2 * logLikelihood;
```

**Key Logic**:
```csharp
var aic = CalculateAIC(model, data);
var bic = CalculateBIC(model, data);

// Calculate for null model (baseline)
var aicNull = CalculateAIC(nullModel, data);
var bicNull = CalculateBIC(nullModel, data);

// Model is better than baseline
if (aic < aicNull && bic < bicNull)
{
    // Check if improvement is substantial
    var aicImprovement = aicNull - aic;
    var bicImprovement = bicNull - bic;

    if (aicImprovement > SignificantThreshold && bicImprovement > SignificantThreshold)
        return FitType.GoodFit;
    else
        return FitType.Moderate;
}

// Model is worse than baseline
if (aic > aicNull && bic > bicNull)
    return FitType.Overfit;  // Complexity not justified

return FitType.Underfit;
```

**Interpreting Differences**:
- ΔIC < 2: Negligible difference
- 2 ≤ ΔIC < 6: Moderate support
- 6 ≤ ΔIC < 10: Strong support
- ΔIC ≥ 10: Very strong support

**FitTypes Returned**:
- `GoodFit` (significant improvement over baseline)
- `Moderate` (slight improvement)
- `Overfit` (complexity not justified)
- `Underfit` (worse than baseline)

---

### 3. AdaptiveFitDetector
**Location**: `C:\Users\cheat\source\repos\AiDotNet\src\FitDetectors\AdaptiveFitDetector.cs`

**What it does**:
- Learns optimal thresholds from data
- Adapts to data distribution characteristics
- Uses historical performance to refine detection

**Key Logic**:
```csharp
// Initialize with default thresholds
var thresholds = new AdaptiveThresholds
{
    OverfitThreshold = 0.1,
    UnderfitThreshold = 0.7,
    GoodFitThreshold = 0.9
};

// Analyze data distribution
var dataStats = AnalyzeDataDistribution(evalData);

// Adjust thresholds based on characteristics
if (dataStats.IsHighDimensional)
{
    // Relax thresholds for high-dimensional data
    thresholds.OverfitThreshold *= 1.5;
    thresholds.GoodFitThreshold *= 0.9;
}

if (dataStats.IsNoisy)
{
    // Be more lenient with noisy data
    thresholds.GoodFitThreshold *= 0.85;
}

if (dataStats.SampleSize < 100)
{
    // Stricter thresholds for small samples
    thresholds.OverfitThreshold *= 0.8;
}

// Use adapted thresholds for detection
return DetectFitWithThresholds(evalData, thresholds);
```

**Adaptation Strategies**:

1. **Data-Driven**:
   - Analyze data noise level
   - Consider sample size
   - Account for dimensionality

2. **Performance-Based**:
   - Track historical fit results
   - Learn which thresholds work best
   - Update thresholds over time

3. **Cross-Validation**:
   - Use CV to estimate optimal thresholds
   - Test multiple threshold combinations
   - Select based on CV performance

**Key Components**:
```csharp
// Analyze data characteristics
var noise = EstimateNoiseLevel(data);
var dimensionality = data.Features.Columns;
var sampleSize = data.Features.Rows;
var complexity = EstimateComplexity(model);

// Adapt thresholds
var adaptedThresholds = AdaptThresholds(
    noise, dimensionality, sampleSize, complexity);

// Detect fit with adapted thresholds
var fitType = DetermineFitType(evalData, adaptedThresholds);
```

**FitTypes Returned**: Any (uses adapted thresholds)

---

### 4. HybridFitDetector
**Location**: `C:\Users\cheat\source\repos\AiDotNet\src\FitDetectors\HybridFitDetector.cs`

**What it does**:
- Intelligently selects detection strategy based on context
- Combines strengths of multiple approaches
- Uses decision tree to route to appropriate detector

**Key Logic**:
```csharp
// Analyze problem characteristics
var problemType = ClassifyProblem(evalData);

IFitDetector<T, TInput, TOutput> selectedDetector;

switch (problemType)
{
    case ProblemType.TimeSeries:
        // Use autocorrelation detector for time series
        selectedDetector = new AutocorrelationFitDetector<T, TInput, TOutput>();
        break;

    case ProblemType.HighDimensional:
        // Use feature importance for high-dimensional data
        selectedDetector = new FeatureImportanceFitDetector<T, TInput, TOutput>();
        break;

    case ProblemType.SmallSample:
        // Use bootstrap for small samples
        selectedDetector = new BootstrapFitDetector<T, TInput, TOutput>();
        break;

    case ProblemType.Classification:
        // Use confusion matrix for classification
        selectedDetector = new ConfusionMatrixFitDetector<T, TInput, TOutput>();
        break;

    case ProblemType.Probabilistic:
        // Use Bayesian or calibration for probabilistic models
        selectedDetector = new BayesianFitDetector<T, TInput, TOutput>();
        break;

    default:
        // Fall back to ensemble
        selectedDetector = new EnsembleFitDetector<T, TInput, TOutput>();
        break;
}

// Use selected detector
var result = selectedDetector.DetectFit(evalData);

// Enhance with hybrid-specific recommendations
result.Recommendations.Add($"Strategy selected: {problemType}");
return result;
```

**Problem Classification**:
```csharp
var problemType = ProblemType.General;

// Check for time series (autocorrelated residuals)
if (HasTemporalDependence(data))
    problemType = ProblemType.TimeSeries;

// Check for high dimensionality (p > n or p > 100)
else if (data.Features.Columns > data.Features.Rows ||
         data.Features.Columns > 100)
    problemType = ProblemType.HighDimensional;

// Check for small sample (n < 30)
else if (data.Features.Rows < 30)
    problemType = ProblemType.SmallSample;

// Check for classification (discrete outputs)
else if (IsClassification(data))
    problemType = ProblemType.Classification;

// Check for probabilistic model
else if (HasProbabilities(data))
    problemType = ProblemType.Probabilistic;

return problemType;
```

**Hybrid Strategies**:
1. **Contextual Selection**: Choose detector based on problem type
2. **Sequential Application**: Apply multiple detectors in sequence
3. **Cascading**: Use simple detectors first, complex ones if needed
4. **Weighted Combination**: Combine results from multiple detectors

**FitTypes Returned**: Depends on selected detector

---

## What's MISSING (Test Coverage)

### Current State
- **0% test coverage** for all Advanced FitDetectors
- No tests for Bayesian metric calculations (DIC, WAIC, LOO)
- No tests for information criteria (AIC, BIC)
- No tests for adaptive threshold learning
- No tests for hybrid detector routing logic

### Required Test Coverage

#### 1. BayesianFitDetector Tests
- **DIC calculation**: Verify against known values
- **WAIC calculation**: Match expected results
- **LOO calculation**: Compare to actual LOO-CV
- **Metric agreement**: All three metrics align on clear cases
- **Metric disagreement**: Handle inconsistent metrics gracefully

#### 2. InformationCriteriaFitDetector Tests
- **AIC calculation**: Verify formula
- **BIC calculation**: Verify formula
- **Baseline comparison**: Model vs. null model
- **Interpretation**: ΔIC thresholds work correctly
- **Edge cases**: Perfect fit, random predictions

#### 3. AdaptiveFitDetector Tests
- **Threshold adaptation**: High-dimensional data relaxes thresholds
- **Noise handling**: Noisy data gets lenient thresholds
- **Sample size**: Small samples get strict thresholds
- **Learning**: Historical performance improves thresholds
- **Convergence**: Thresholds stabilize over time

#### 4. HybridFitDetector Tests
- **Problem classification**: Correctly identifies problem types
- **Detector selection**: Routes to appropriate detector
- **Time series routing**: Uses AutocorrelationFitDetector
- **Classification routing**: Uses ConfusionMatrixFitDetector
- **Fallback**: Uses EnsembleFitDetector for unknown types

---

## Step-by-Step Test Implementation

### Step 1: BayesianFitDetector Tests

```csharp
public class BayesianFitDetectorTests
{
    [Fact]
    public void CalculateDIC_KnownModel_MatchesExpected()
    {
        // Arrange: Simple linear model with known DIC
        var model = CreateLinearModel();
        var data = CreateLinearData();

        // Expected DIC calculation:
        // k = 2 parameters (slope, intercept)
        // logL = log-likelihood
        // DIC = -2*logL + 2*k

        var expectedDIC = CalculateExpectedDIC(model, data);

        // Act
        var detector = new BayesianFitDetector<double, double[], double>();
        var evalData = CreateEvalDataFromModel(model, data);
        var result = detector.DetectFit(evalData);

        // Assert: DIC should be in AdditionalInfo
        Assert.True(result.AdditionalInfo.ContainsKey("DIC"));
        var actualDIC = (double)result.AdditionalInfo["DIC"];
        Assert.Equal(expectedDIC, actualDIC, precision: 1);
    }

    [Fact]
    public void CalculateWAIC_SimpleCase_Reasonable()
    {
        // Arrange
        var detector = new BayesianFitDetector<double, double[], double>();
        var evalData = CreateSimpleLinearData();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: WAIC should be positive and finite
        Assert.True(result.AdditionalInfo.ContainsKey("WAIC"));
        var waic = (double)result.AdditionalInfo["WAIC"];
        Assert.True(waic > 0);
        Assert.True(Double.IsFinite(waic));
    }

    [Fact]
    public void DetectFit_AllMetricsLow_GoodFit()
    {
        // Arrange: Well-fitting model
        var detector = new BayesianFitDetector<double, double[], double>();
        var evalData = CreateWellFittingModel();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.GoodFit, result.FitType);
    }

    [Fact]
    public void DetectFit_AllMetricsHigh_Overfit()
    {
        // Arrange: Overfitting model (too complex)
        var detector = new BayesianFitDetector<double, double[], double>();
        var evalData = CreateOverfittingModel();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.Overfit, result.FitType);
    }

    [Fact]
    public void DetectFit_InconsistentMetrics_Unstable()
    {
        // Arrange: DIC says good, WAIC says bad
        var detector = new BayesianFitDetector<double, double[], double>();
        var evalData = CreateInconsistentMetricsData();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(FitType.Unstable, result.FitType);
    }

    [Fact]
    public void LOO_VsCrossValidation_Similar()
    {
        // Arrange: Compare LOO approximation to actual LOO-CV
        var detector = new BayesianFitDetector<double, double[], double>();
        var data = CreateSmallDataset(n: 20);

        // Calculate actual LOO-CV manually
        var actualLOO = CalculateActualLOOCV(data);

        // Act: Get LOO from detector
        var evalData = CreateEvalDataFromData(data);
        var result = detector.DetectFit(evalData);
        var approximateLOO = (double)result.AdditionalInfo["LOO"];

        // Assert: Should be close (within 10%)
        Assert.InRange(approximateLOO, actualLOO * 0.9, actualLOO * 1.1);
    }
}
```

---

### Step 2: InformationCriteriaFitDetector Tests

```csharp
public class InformationCriteriaFitDetectorTests
{
    [Fact]
    public void CalculateAIC_KnownModel_MatchesFormula()
    {
        // Arrange: Model with k=3 parameters
        var model = CreateModelWithKParameters(k: 3);
        var data = CreateData(n: 100);
        var logL = CalculateLogLikelihood(model, data);

        // Expected: AIC = 2*k - 2*ln(L) = 2*3 - 2*logL = 6 - 2*logL
        var expectedAIC = 6 - 2 * logL;

        // Act
        var detector = new InformationCriteriaFitDetector<double, double[], double>();
        var evalData = CreateEvalData(model, data);
        var result = detector.DetectFit(evalData);

        // Assert
        var actualAIC = (double)result.AdditionalInfo["AIC"];
        Assert.Equal(expectedAIC, actualAIC, precision: 2);
    }

    [Fact]
    public void CalculateBIC_KnownModel_MatchesFormula()
    {
        // Arrange: Model with k=3 parameters, n=100 data points
        var model = CreateModelWithKParameters(k: 3);
        var data = CreateData(n: 100);
        var logL = CalculateLogLikelihood(model, data);

        // Expected: BIC = k*ln(n) - 2*ln(L) = 3*ln(100) - 2*logL
        var expectedBIC = 3 * Math.Log(100) - 2 * logL;

        // Act
        var detector = new InformationCriteriaFitDetector<double, double[], double>();
        var evalData = CreateEvalData(model, data);
        var result = detector.DetectFit(evalData);

        // Assert
        var actualBIC = (double)result.AdditionalInfo["BIC"];
        Assert.Equal(expectedBIC, actualBIC, precision: 2);
    }

    [Theory]
    [InlineData(10, FitType.GoodFit)]      // ΔAIC = 10 (very strong)
    [InlineData(6, FitType.GoodFit)]       // ΔAIC = 6 (strong)
    [InlineData(2, FitType.Moderate)]      // ΔAIC = 2 (moderate)
    [InlineData(0, FitType.Underfit)]      // ΔAIC = 0 (no improvement)
    [InlineData(-5, FitType.Overfit)]      // ΔAIC = -5 (worse than baseline)
    public void DetectFit_DifferentAICDifferences_CorrectFitType(
        double aicDifference,
        FitType expectedFitType)
    {
        // Arrange: Model with specified AIC difference from baseline
        var detector = new InformationCriteriaFitDetector<double, double[], double>();
        var evalData = CreateEvalDataWithAICDifference(aicDifference);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert
        Assert.Equal(expectedFitType, result.FitType);
    }

    [Fact]
    public void DetectFit_PerfectFit_BetterThanBaseline()
    {
        // Arrange: Perfect model (R2 = 1.0)
        var detector = new InformationCriteriaFitDetector<double, double[], double>();
        var evalData = CreatePerfectFitData();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: Should be significantly better than null model
        Assert.Equal(FitType.GoodFit, result.FitType);
    }

    [Fact]
    public void DetectFit_RandomPredictions_WorseThanBaseline()
    {
        // Arrange: Model makes random predictions
        var detector = new InformationCriteriaFitDetector<double, double[], double>();
        var evalData = CreateRandomPredictionsData();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: Should be worse than null model
        Assert.True(result.FitType == FitType.Overfit ||
                    result.FitType == FitType.Underfit);
    }
}
```

---

### Step 3: AdaptiveFitDetector Tests

```csharp
public class AdaptiveFitDetectorTests
{
    [Fact]
    public void AdaptThresholds_HighDimensional_RelaxesThresholds()
    {
        // Arrange: High-dimensional data (p >> n)
        var detector = new AdaptiveFitDetector<double, double[], double>();
        var evalData = CreateHighDimensionalData(features: 1000, samples: 100);

        // Get initial thresholds
        var initialThresholds = detector.GetCurrentThresholds();

        // Act
        var result = detector.DetectFit(evalData);

        // Get adapted thresholds
        var adaptedThresholds = detector.GetCurrentThresholds();

        // Assert: Thresholds should be relaxed for high-dimensional data
        Assert.True(adaptedThresholds.OverfitThreshold > initialThresholds.OverfitThreshold);
    }

    [Fact]
    public void AdaptThresholds_NoisyData_LenientThresholds()
    {
        // Arrange: Data with high noise
        var detector = new AdaptiveFitDetector<double, double[], double>();
        var evalData = CreateNoisyData(noiseLevel: 0.5);

        // Act
        var result = detector.DetectFit(evalData);
        var thresholds = detector.GetCurrentThresholds();

        // Assert: GoodFitThreshold should be lower (more lenient)
        Assert.True(thresholds.GoodFitThreshold < 0.9);  // Default is 0.9
    }

    [Fact]
    public void AdaptThresholds_SmallSample_StricterThresholds()
    {
        // Arrange: Small sample size
        var detector = new AdaptiveFitDetector<double, double[], double>();
        var evalData = CreateSmallSampleData(n: 20);

        // Act
        var result = detector.DetectFit(evalData);
        var thresholds = detector.GetCurrentThresholds();

        // Assert: OverfitThreshold should be stricter (lower)
        Assert.True(thresholds.OverfitThreshold < 0.1);  // Default is 0.1
    }

    [Fact]
    public void Learning_MultipleRuns_ThresholdsConverge()
    {
        // Arrange: Same type of data multiple times
        var detector = new AdaptiveFitDetector<double, double[], double>(
            new AdaptiveFitDetectorOptions { EnableLearning = true });

        var thresholdHistory = new List<double>();

        // Act: Run 10 times on similar data
        for (int i = 0; i < 10; i++)
        {
            var evalData = CreateSimilarData(seed: i);
            detector.DetectFit(evalData);
            thresholdHistory.Add(detector.GetCurrentThresholds().OverfitThreshold);
        }

        // Assert: Thresholds should stabilize (variance decreases)
        var earlyVariance = Variance(thresholdHistory.Take(3));
        var lateVariance = Variance(thresholdHistory.Skip(7).Take(3));
        Assert.True(lateVariance < earlyVariance);
    }

    [Fact]
    public void DetectFit_AdaptedThresholds_MoreAccurate()
    {
        // Arrange: Data that would be misclassified with fixed thresholds
        var fixedDetector = new DefaultFitDetector<double, double[], double>();
        var adaptiveDetector = new AdaptiveFitDetector<double, double[], double>();

        var evalData = CreateBorderlineCase();

        // Act
        var fixedResult = fixedDetector.DetectFit(evalData);
        var adaptiveResult = adaptiveDetector.DetectFit(evalData);

        // Assert: Adaptive should have higher confidence
        Assert.True(adaptiveResult.ConfidenceLevel > fixedResult.ConfidenceLevel);
    }
}
```

---

### Step 4: HybridFitDetector Tests

```csharp
public class HybridFitDetectorTests
{
    [Fact]
    public void ClassifyProblem_TimeSeries_RoutesToAutocorrelation()
    {
        // Arrange: Time series data
        var detector = new HybridFitDetector<double, double[], double>();
        var evalData = CreateTimeSeriesData();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: Should mention autocorrelation in recommendations
        Assert.Contains(result.Recommendations,
            r => r.Contains("autocorrelation", StringComparison.OrdinalIgnoreCase) ||
                 r.Contains("time series", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void ClassifyProblem_HighDimensional_RoutesToFeatureImportance()
    {
        // Arrange: High-dimensional data
        var detector = new HybridFitDetector<double, double[], double>();
        var evalData = CreateHighDimensionalData(features: 200, samples: 50);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: Should mention feature importance or selection
        Assert.Contains(result.Recommendations,
            r => r.Contains("feature", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void ClassifyProblem_SmallSample_RoutesToBootstrap()
    {
        // Arrange: Small sample
        var detector = new HybridFitDetector<double, double[], double>();
        var evalData = CreateSmallSampleData(n: 25);

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: Should mention bootstrap or resampling
        Assert.Contains(result.Recommendations,
            r => r.Contains("bootstrap", StringComparison.OrdinalIgnoreCase) ||
                 r.Contains("resample", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void ClassifyProblem_Classification_RoutesToConfusionMatrix()
    {
        // Arrange: Classification data
        var detector = new HybridFitDetector<double, double[], int>();
        var evalData = CreateClassificationData();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: Should mention classification metrics
        Assert.Contains(result.Recommendations,
            r => r.Contains("accuracy", StringComparison.OrdinalIgnoreCase) ||
                 r.Contains("precision", StringComparison.OrdinalIgnoreCase) ||
                 r.Contains("recall", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void ClassifyProblem_General_RoutesToEnsemble()
    {
        // Arrange: General case (no special characteristics)
        var detector = new HybridFitDetector<double, double[], double>();
        var evalData = CreateGeneralData();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: Should use ensemble (default)
        Assert.NotNull(result);
        // Check that multiple detectors were consulted (if exposed)
    }

    [Theory]
    [InlineData(ProblemType.TimeSeries, typeof(AutocorrelationFitDetector<,,>))]
    [InlineData(ProblemType.HighDimensional, typeof(FeatureImportanceFitDetector<,,>))]
    [InlineData(ProblemType.SmallSample, typeof(BootstrapFitDetector<,,>))]
    [InlineData(ProblemType.Classification, typeof(ConfusionMatrixFitDetector<,,>))]
    public void SelectDetector_DifferentProblemTypes_CorrectDetectorType(
        ProblemType problemType,
        Type expectedDetectorType)
    {
        // Arrange
        var detector = new HybridFitDetector<double, double[], double>();
        var evalData = CreateDataForProblemType(problemType);

        // Act
        var selectedDetector = detector.SelectDetectorForProblem(evalData);

        // Assert: Should select correct detector type
        Assert.True(selectedDetector.GetType().GetGenericTypeDefinition() == expectedDetectorType);
    }

    [Fact]
    public void Recommendations_IncludeStrategySelected()
    {
        // Arrange
        var detector = new HybridFitDetector<double, double[], double>();
        var evalData = CreateTimeSeriesData();

        // Act
        var result = detector.DetectFit(evalData);

        // Assert: Should explain which strategy was used
        Assert.Contains(result.Recommendations,
            r => r.Contains("strategy", StringComparison.OrdinalIgnoreCase));
    }
}
```

---

## Testing Strategy

### 1. Bayesian Testing
- Compare calculations to R package `loo`
- Use simple models with known Bayesian metrics
- Test on synthetic data with known properties

### 2. Information Criteria Testing
- Verify against textbook examples
- Test on models of different complexity
- Compare AIC/BIC rankings to ground truth

### 3. Adaptive Testing
- Track threshold evolution over time
- Test convergence on stable data
- Test adaptation to different data types

### 4. Hybrid Testing
- Test all routing paths (time series, classification, etc.)
- Verify fallback to ensemble
- Test that selected detector is appropriate

### 5. Coverage Goals
- **Minimum**: 80% code coverage
- **Target**: 90% including all routing logic
- **Focus**: Problem classification, detector selection

---

## Common Pitfalls

### 1. Bayesian Metrics Require Samples
**Problem**: DIC/WAIC/LOO need posterior samples, not just point estimates

**Solution**: Either generate posterior samples or use approximations

### 2. Information Criteria Need Likelihood
**Problem**: Not all models provide log-likelihood

**Solution**: Calculate from residuals:
```csharp
logL = -n/2 * (log(2π) + log(MSE) + 1)
```

### 3. Adaptive Learning Overfits to Data
**Problem**: Thresholds adapt too much to specific dataset

**Solution**: Use exponential moving average for stability

### 4. Hybrid Routing Ambiguity
**Problem**: Data fits multiple problem types

**Solution**: Use priority order or combine multiple detectors

### 5. Cross-Detector Consistency
**Problem**: Hybrid may give different results than constituent detectors

**Solution**: This is expected - document behavior clearly

---

## Next Steps

1. Read all four detector implementations completely
2. Research Bayesian metrics (DIC, WAIC, LOO)
3. Understand information criteria mathematics
4. Implement test data generators for each scenario
5. Create comprehensive test suite
6. Document any theoretical issues or limitations found

---

## Additional Resources

- **Bayesian Model Comparison**: Gelman et al., "Bayesian Data Analysis"
- **DIC, WAIC, LOO**: https://arxiv.org/abs/1507.04544
- **AIC/BIC**: Burnham & Anderson, "Model Selection and Multimodel Inference"
- **Information Criteria**: https://en.wikipedia.org/wiki/Akaike_information_criterion
- **R loo package**: https://mc-stan.org/loo/
- **Adaptive Methods**: https://en.wikipedia.org/wiki/Adaptive_algorithm

---

## Summary

**Advanced FitDetectors** represent the cutting edge of fit detection:
- **Bayesian**: Most principled approach, accounts for uncertainty
- **Information Criteria**: Balance fit and complexity elegantly
- **Adaptive**: Learns optimal thresholds from data
- **Hybrid**: Intelligently selects best approach for each problem

These detectors require the most sophisticated testing due to their mathematical complexity and adaptive nature. Focus on correctness of calculations, proper routing logic, and robust handling of edge cases.
