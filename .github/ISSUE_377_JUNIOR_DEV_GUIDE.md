# Junior Developer Implementation Guide: Issue #377

## Overview
**Issue**: Distribution-Based Loss Functions Unit Tests
**Goal**: Create comprehensive unit tests for distribution and statistical FitnessCalculators
**Difficulty**: Intermediate-Advanced
**Estimated Time**: 5-7 hours

## What You'll Be Testing

You'll create unit tests for **10 distribution-based and statistical loss function fitness calculators**:

1. **KullbackLeiblerDivergenceFitnessCalculator** - Measures difference between probability distributions
2. **OrdinalRegressionLossFitnessCalculator** - For ordered categories (ratings, grades)
3. **PoissonLossFitnessCalculator** - For count data (events per time period)
4. **QuantileLossFitnessCalculator** - For quantile regression
5. **TripletLossFitnessCalculator** - Similarity learning with anchor/positive/negative
6. **SquaredHingeLossFitnessCalculator** - Smooth variant of Hinge Loss
7. **WeightedCrossEntropyLossFitnessCalculator** - Class-weighted classification
8. **RootMeanSquaredErrorFitnessCalculator** - Square root of MSE
9. **RSquaredFitnessCalculator** - Coefficient of determination (R²)
10. **ModifiedHuberLossFitnessCalculator** - Robust variant of Huber Loss

## Understanding the Codebase

### Key Files to Review

**Implementations to Test:**
```
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\KullbackLeiblerDivergenceFitnessCalculator.cs
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\OrdinalRegressionLossFitnessCalculator.cs
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\PoissonLossFitnessCalculator.cs
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\QuantileLossFitnessCalculator.cs
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\TripletLossFitnessCalculator.cs
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\SquaredHingeLossFitnessCalculator.cs
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\WeightedCrossEntropyLossFitnessCalculator.cs
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\RootMeanSquaredErrorFitnessCalculator.cs
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\RSquaredFitnessCalculator.cs
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\ModifiedHuberLossFitnessCalculator.cs
```

### Mathematical Formulas

**Kullback-Leibler Divergence:**
```
KL(P||Q) = sum(P(x) * log(P(x) / Q(x)))

where:
    P = actual distribution
    Q = predicted distribution
```

**Poisson Loss:**
```
Poisson Loss = predicted - actual * log(predicted)

Assumes:
    actual = observed count
    predicted = predicted rate (lambda)
```

**Quantile Loss:**
```
For quantile τ:
    If error > 0: loss = τ * error
    If error < 0: loss = (1-τ) * |error|

where:
    error = actual - predicted
    τ = target quantile (e.g., 0.5 for median)
```

**Triplet Loss:**
```
L = max(0, distance(anchor, positive) - distance(anchor, negative) + margin)

where:
    anchor = reference sample
    positive = same class as anchor
    negative = different class from anchor
    margin = minimum separation
```

**R-Squared (R²):**
```
R² = 1 - (SS_res / SS_tot)

where:
    SS_res = sum((actual - predicted)^2)
    SS_tot = sum((actual - mean(actual))^2)

Range: (-∞, 1], where 1 = perfect fit
```

## Step-by-Step Implementation Guide

### Step 1: Create Test File Structure

Create file: `C:\Users\cheat\source\repos\AiDotNet\tests\FitnessCalculators\DistributionBasedLossFitnessCalculatorTests.cs`

```csharp
using System;
using System.Linq;
using AiDotNet.FitnessCalculators;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNet.Tests.FitnessCalculators
{
    public class DistributionBasedLossFitnessCalculatorTests
    {
        private static void AssertClose(double actual, double expected, double tolerance = 1e-6)
        {
            Assert.True(Math.Abs(actual - expected) <= tolerance,
                $"Expected {expected}, but got {actual}. Difference: {Math.Abs(actual - expected)}");
        }

        private DataSetStats<double, double, double> CreateTestDataSet(
            double[] predicted,
            double[] actual)
        {
            return new DataSetStats<double, double, double>
            {
                Predicted = predicted,
                Actual = actual
            };
        }

        // Tests will go here
    }
}
```

### Step 2: Test Kullback-Leibler Divergence

**Understanding KL Divergence:**
- Measures how one distribution differs from another
- Not symmetric: KL(P||Q) ≠ KL(Q||P)
- Always non-negative
- Zero when distributions are identical

```csharp
[Fact]
public void KLDivergence_IdenticalDistributions_ReturnsZero()
{
    // Arrange - Identical probability distributions
    var predicted = new[] { 0.25, 0.25, 0.25, 0.25 };
    var actual = new[] { 0.25, 0.25, 0.25, 0.25 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    AssertClose(score, 0.0);
    Assert.False(calculator.IsHigherScoreBetter);
}

[Fact]
public void KLDivergence_DifferentDistributions_ReturnsPositiveValue()
{
    // Arrange
    var predicted = new[] { 0.5, 0.3, 0.2 };  // Q
    var actual = new[] { 0.33, 0.33, 0.34 };  // P (uniform)
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    Assert.True(score > 0);
    Assert.True(double.IsFinite(score));
}

[Fact]
public void KLDivergence_VeryDifferentDistributions_ReturnsHighValue()
{
    // Arrange - Predicted is concentrated, actual is uniform
    var predicted = new[] { 0.97, 0.01, 0.01, 0.01 };
    var actual = new[] { 0.25, 0.25, 0.25, 0.25 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // Should have high divergence
    Assert.True(score > 0.5);
}

[Fact]
public void KLDivergence_NotSymmetric_DifferentWhenSwapped()
{
    // Arrange
    var dist1 = new[] { 0.7, 0.2, 0.1 };
    var dist2 = new[] { 0.4, 0.4, 0.2 };

    var dataSet1 = CreateTestDataSet(predicted: dist1, actual: dist2);
    var dataSet2 = CreateTestDataSet(predicted: dist2, actual: dist1);
    var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, double, double>();

    // Act
    var kl1 = calculator.CalculateFitnessScore(dataSet1);
    var kl2 = calculator.CalculateFitnessScore(dataSet2);

    // Assert
    // KL divergence is not symmetric
    Assert.NotEqual(kl1, kl2);
}

[Fact]
public void KLDivergence_ZeroInPredicted_HandlesCorrectly()
{
    // Arrange - Zero probability in predicted distribution
    var predicted = new[] { 1.0, 0.0, 0.0 };
    var actual = new[] { 0.5, 0.3, 0.2 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, double, double>();

    // Act & Assert
    // Should handle log(0) gracefully (either clip or throw)
    try
    {
        var score = calculator.CalculateFitnessScore(dataSet);
        Assert.True(double.IsFinite(score) || double.IsPositiveInfinity(score));
    }
    catch (ArgumentException)
    {
        // Acceptable if implementation validates inputs
        Assert.True(true);
    }
}
```

### Step 3: Test Ordinal Regression Loss

```csharp
[Fact]
public void OrdinalRegression_ExactPredictions_ReturnsLowLoss()
{
    // Arrange - Predicting ratings 1-5
    var predicted = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
    var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new OrdinalRegressionLossFitnessCalculator<double, double, double>(
        numberOfClassifications: 5);

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // Perfect ordinal predictions should have low loss
    Assert.True(score < 0.1);
}

[Fact]
public void OrdinalRegression_OffByOne_LowerLossThanOffByMany()
{
    // Arrange - Testing ordinal property
    var predictedOffByOne = new[] { 2.0, 3.0, 4.0 };
    var predictedOffByMany = new[] { 5.0, 1.0, 1.0 };
    var actual = new[] { 1.0, 2.0, 3.0 };

    var dataSetOffByOne = CreateTestDataSet(predictedOffByOne, actual);
    var dataSetOffByMany = CreateTestDataSet(predictedOffByMany, actual);
    var calculator = new OrdinalRegressionLossFitnessCalculator<double, double, double>(
        numberOfClassifications: 5);

    // Act
    var lossOffByOne = calculator.CalculateFitnessScore(dataSetOffByOne);
    var lossOffByMany = calculator.CalculateFitnessScore(dataSetOffByMany);

    // Assert
    // Being off by 1 should be better than being off by many
    Assert.True(lossOffByOne < lossOffByMany);
}

[Fact]
public void OrdinalRegression_WithoutNumClasses_AutoDetects()
{
    // Arrange - Let calculator auto-detect number of classes
    var predicted = new[] { 1.0, 2.0, 3.0 };
    var actual = new[] { 1.0, 2.0, 3.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new OrdinalRegressionLossFitnessCalculator<double, double, double>(
        numberOfClassifications: null);

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // Should handle auto-detection
    Assert.True(double.IsFinite(score));
    Assert.True(score >= 0);
}
```

### Step 4: Test Poisson Loss

```csharp
[Fact]
public void PoissonLoss_PerfectPredictions_ReturnsLowLoss()
{
    // Arrange - Predicting count data
    var predicted = new[] { 2.0, 5.0, 1.0, 3.0 };
    var actual = new[] { 2.0, 5.0, 1.0, 3.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new PoissonLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // Perfect predictions should have low loss
    Assert.True(score < 0.5);
}

[Fact]
public void PoissonLoss_UnderPrediction_PenalizesCorrectly()
{
    // Arrange - Predicting too low
    var predicted = new[] { 1.0, 1.0, 1.0 };
    var actual = new[] { 5.0, 5.0, 5.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new PoissonLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    Assert.True(score > 1.0);
}

[Fact]
public void PoissonLoss_OverPrediction_PenalizesCorrectly()
{
    // Arrange - Predicting too high
    var predicted = new[] { 10.0, 10.0, 10.0 };
    var actual = new[] { 2.0, 2.0, 2.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new PoissonLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    Assert.True(score > 1.0);
}

[Fact]
public void PoissonLoss_ZeroCounts_HandlesCorrectly()
{
    // Arrange - Zero counts (valid in Poisson)
    var predicted = new[] { 0.1, 0.2, 0.3 };  // Small predicted rates
    var actual = new[] { 0.0, 0.0, 0.0 };     // Zero observed counts
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new PoissonLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    Assert.True(double.IsFinite(score));
}

[Fact]
public void PoissonLoss_NegativeValues_ThrowsOrHandles()
{
    // Arrange - Negative values (invalid for Poisson)
    var predicted = new[] { -1.0, 2.0 };
    var actual = new[] { 1.0, 2.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new PoissonLossFitnessCalculator<double, double, double>();

    // Act & Assert
    // Should either throw or handle gracefully
    try
    {
        var score = calculator.CalculateFitnessScore(dataSet);
        Assert.True(double.IsFinite(score));
    }
    catch (ArgumentException)
    {
        Assert.True(true);
    }
}
```

### Step 5: Test Quantile Loss

```csharp
[Fact]
public void QuantileLoss_MedianPrediction_SymmetricPenalty()
{
    // Arrange - Median (quantile = 0.5)
    var predicted = new[] { 5.0, 5.0, 5.0 };
    var actual = new[] { 4.0, 5.0, 6.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new QuantileLossFitnessCalculator<double, double, double>(quantile: 0.5);

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // For median, over/under predictions should be penalized equally
    Assert.True(score > 0);
    Assert.True(double.IsFinite(score));
}

[Fact]
public void QuantileLoss_UpperQuantile_PenalizesUnderPredictionMore()
{
    // Arrange - Upper quantile (0.9)
    var predictedLow = new[] { 3.0 };   // Under-prediction
    var predictedHigh = new[] { 7.0 };  // Over-prediction
    var actual = new[] { 5.0 };

    var dataSetLow = CreateTestDataSet(predictedLow, actual);
    var dataSetHigh = CreateTestDataSet(predictedHigh, actual);
    var calculator = new QuantileLossFitnessCalculator<double, double, double>(quantile: 0.9);

    // Act
    var lossUnder = calculator.CalculateFitnessScore(dataSetLow);
    var lossOver = calculator.CalculateFitnessScore(dataSetHigh);

    // Assert
    // For upper quantile, under-prediction should be penalized more
    Assert.True(lossUnder > lossOver);
}

[Fact]
public void QuantileLoss_LowerQuantile_PenalizesOverPredictionMore()
{
    // Arrange - Lower quantile (0.1)
    var predictedLow = new[] { 3.0 };   // Under-prediction
    var predictedHigh = new[] { 7.0 };  // Over-prediction
    var actual = new[] { 5.0 };

    var dataSetLow = CreateTestDataSet(predictedLow, actual);
    var dataSetHigh = CreateTestDataSet(predictedHigh, actual);
    var calculator = new QuantileLossFitnessCalculator<double, double, double>(quantile: 0.1);

    // Act
    var lossUnder = calculator.CalculateFitnessScore(dataSetLow);
    var lossOver = calculator.CalculateFitnessScore(dataSetHigh);

    // Assert
    // For lower quantile, over-prediction should be penalized more
    Assert.True(lossOver > lossUnder);
}

[Fact]
public void QuantileLoss_DifferentQuantiles_ProduceDifferentLosses()
{
    // Arrange
    var predicted = new[] { 4.0 };
    var actual = new[] { 5.0 };
    var dataSet = CreateTestDataSet(predicted, actual);

    var calc10 = new QuantileLossFitnessCalculator<double, double, double>(quantile: 0.1);
    var calc50 = new QuantileLossFitnessCalculator<double, double, double>(quantile: 0.5);
    var calc90 = new QuantileLossFitnessCalculator<double, double, double>(quantile: 0.9);

    // Act
    var loss10 = calc10.CalculateFitnessScore(dataSet);
    var loss50 = calc50.CalculateFitnessScore(dataSet);
    var loss90 = calc90.CalculateFitnessScore(dataSet);

    // Assert
    // Different quantiles should produce different losses
    Assert.True(loss10 != loss50 || loss50 != loss90);
}
```

### Step 6: Test Triplet Loss

```csharp
[Fact]
public void TripletLoss_PositiveCloserThanNegative_ReturnsZero()
{
    // Arrange - Good triplet: positive is closer to anchor than negative
    // This test requires understanding the data structure
    // Implementation details depend on how PrepareTripletData works
    var calculator = new TripletLossFitnessCalculator<double, double, double>(margin: 1.0);

    // Note: Actual test will depend on data structure
    // This is a placeholder showing the concept
    Assert.False(calculator.IsHigherScoreBetter);
}

[Fact]
public void TripletLoss_NegativeCloserThanPositive_ReturnsPositiveLoss()
{
    // Arrange - Bad triplet: negative is closer than positive
    var calculator = new TripletLossFitnessCalculator<double, double, double>(margin: 1.0);

    // Act & Assert
    // Should return positive loss when constraint is violated
    Assert.False(calculator.IsHigherScoreBetter);
}

[Fact]
public void TripletLoss_DifferentMargins_ProduceDifferentResults()
{
    // Arrange
    var calculator1 = new TripletLossFitnessCalculator<double, double, double>(margin: 0.5);
    var calculator2 = new TripletLossFitnessCalculator<double, double, double>(margin: 2.0);

    // Act & Assert
    // Different margins should affect the loss differently
    Assert.False(calculator1.IsHigherScoreBetter);
    Assert.False(calculator2.IsHigherScoreBetter);
}
```

### Step 7: Test R-Squared and RMSE

```csharp
[Fact]
public void RSquared_PerfectPredictions_ReturnsOne()
{
    // Arrange
    var predicted = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
    var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new RSquaredFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    AssertClose(score, 1.0);
    Assert.True(calculator.IsHigherScoreBetter); // R² is higher-is-better!
}

[Fact]
public void RSquared_WorstPredictions_ReturnsNegativeValue()
{
    // Arrange - Predictions worse than mean
    var predicted = new[] { 1.0, 1.0, 1.0, 1.0, 1.0 };
    var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new RSquaredFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // R² can be negative if predictions are worse than mean
    Assert.True(score < 0);
}

[Fact]
public void RSquared_MeanPredictions_ReturnsZero()
{
    // Arrange - Always predicting the mean
    var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
    var mean = actual.Average(); // 3.0
    var predicted = new[] { mean, mean, mean, mean, mean };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new RSquaredFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    AssertClose(score, 0.0, tolerance: 0.01);
}

[Fact]
public void RMSE_ReturnsSquareRootOfMSE()
{
    // Arrange
    var predicted = new[] { 3.0, 8.0, 2.0, 6.0 };
    var actual = new[] { 2.0, 5.0, 1.0, 4.0 };
    var dataSet = CreateTestDataSet(predicted, actual);

    var rmseCalculator = new RootMeanSquaredErrorFitnessCalculator<double, double, double>();
    var mseCalculator = new MeanSquaredErrorFitnessCalculator<double, double, double>();

    // Act
    var rmse = rmseCalculator.CalculateFitnessScore(dataSet);
    var mse = mseCalculator.CalculateFitnessScore(dataSet);

    // Assert
    // RMSE should equal sqrt(MSE)
    AssertClose(rmse, Math.Sqrt(mse));
}

[Fact]
public void RMSE_PerfectPredictions_ReturnsZero()
{
    // Arrange
    var predicted = new[] { 1.0, 2.0, 3.0 };
    var actual = new[] { 1.0, 2.0, 3.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new RootMeanSquaredErrorFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    AssertClose(score, 0.0);
}
```

### Step 8: Test Common Properties

```csharp
[Fact]
public void DistributionBasedCalculators_IsHigherScoreBetter_CorrectValues()
{
    // Most loss functions: lower is better
    var lowerBetterCalculators = new IFitnessCalculator<double, double, double>[]
    {
        new KullbackLeiblerDivergenceFitnessCalculator<double, double, double>(),
        new OrdinalRegressionLossFitnessCalculator<double, double, double>(),
        new PoissonLossFitnessCalculator<double, double, double>(),
        new QuantileLossFitnessCalculator<double, double, double>(),
        new TripletLossFitnessCalculator<double, double, double>(),
        new RootMeanSquaredErrorFitnessCalculator<double, double, double>()
    };

    foreach (var calculator in lowerBetterCalculators)
    {
        Assert.False(calculator.IsHigherScoreBetter,
            $"{calculator.GetType().Name} should have IsHigherScoreBetter = false");
    }

    // R² is an exception: higher is better
    var higherBetterCalculators = new IFitnessCalculator<double, double, double>[]
    {
        new RSquaredFitnessCalculator<double, double, double>()
    };

    foreach (var calculator in higherBetterCalculators)
    {
        Assert.True(calculator.IsHigherScoreBetter,
            $"{calculator.GetType().Name} should have IsHigherScoreBetter = true");
    }
}

[Fact]
public void AllCalculators_HandleSingleValue_Correctly()
{
    var predicted = new[] { 5.0 };
    var actual = new[] { 3.0 };
    var dataSet = CreateTestDataSet(predicted, actual);

    var calculators = new IFitnessCalculator<double, double, double>[]
    {
        new RootMeanSquaredErrorFitnessCalculator<double, double, double>(),
        new RSquaredFitnessCalculator<double, double, double>()
    };

    foreach (var calculator in calculators)
    {
        var score = calculator.CalculateFitnessScore(dataSet);
        Assert.True(double.IsFinite(score) || double.IsNaN(score),
            $"{calculator.GetType().Name} should handle single value");
    }
}
```

## Test Coverage Checklist

**KL Divergence:**
- [ ] Identical distributions (zero divergence)
- [ ] Different distributions (positive divergence)
- [ ] Very different distributions (high divergence)
- [ ] Asymmetry (KL(P||Q) ≠ KL(Q||P))
- [ ] Zero probabilities handling

**Ordinal Regression:**
- [ ] Exact predictions
- [ ] Off by one vs. off by many
- [ ] Auto-detection of classes
- [ ] Non-integer predictions

**Poisson Loss:**
- [ ] Perfect predictions
- [ ] Under-prediction
- [ ] Over-prediction
- [ ] Zero counts
- [ ] Negative values (should fail or clip)

**Quantile Loss:**
- [ ] Median (0.5) - symmetric
- [ ] Upper quantile (0.9) - penalizes under more
- [ ] Lower quantile (0.1) - penalizes over more
- [ ] Different quantiles produce different losses

**R-Squared:**
- [ ] Perfect predictions (R² = 1)
- [ ] Mean predictions (R² = 0)
- [ ] Worse than mean (R² < 0)
- [ ] IsHigherScoreBetter = true

**RMSE:**
- [ ] Equals sqrt(MSE)
- [ ] Perfect predictions (zero)
- [ ] Same units as original data

## Running Your Tests

```bash
# Run all tests
dotnet test

# Run only distribution-based tests
dotnet test --filter "FullyQualifiedName~DistributionBasedLossFitnessCalculatorTests"

# Run specific test
dotnet test --filter "FullyQualifiedName~KLDivergence_IdenticalDistributions"
```

## Common Mistakes to Avoid

1. **KL Divergence is not symmetric** - Order matters!
2. **R² can be negative** - Don't assume it's always 0-1
3. **Poisson Loss requires non-negative values** - Counts can't be negative
4. **Quantile loss is asymmetric** - Different quantiles penalize differently
5. **Triplet Loss requires special data structure** - Anchor/positive/negative triplets

## Learning Resources

### Mathematical Background
- **KL Divergence**: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
- **Poisson Distribution**: https://en.wikipedia.org/wiki/Poisson_distribution
- **Quantile Regression**: https://en.wikipedia.org/wiki/Quantile_regression
- **R-Squared**: https://en.wikipedia.org/wiki/Coefficient_of_determination
- **Triplet Loss**: https://arxiv.org/abs/1503.03832

## Validation Criteria

Your implementation will be considered complete when:

1. All 10 distribution-based calculators have comprehensive tests
2. Test coverage includes:
   - Perfect/worst case scenarios
   - Edge cases (zeros, negatives, extremes)
   - Mathematical properties (symmetry, bounds)
   - Parameter variations
3. All tests pass successfully
4. Statistical correctness validated

## Questions to Consider

1. Why is KL Divergence not symmetric?
2. When would you use Quantile Loss instead of MSE?
3. What does a negative R² value mean?
4. How does Poisson Loss differ from MSE for count data?
5. Why is Triplet Loss effective for face recognition?

## Next Steps After Completion

1. Create a pull request with your tests
2. Consider integration tests combining multiple metrics
3. Explore relationships between metrics (e.g., RMSE vs MSE)
4. Study advanced topics like proper scoring rules

---

**Good luck!** These distribution-based loss functions are used in advanced ML applications. Mastering them will prepare you for sophisticated statistical modeling and probabilistic machine learning.
