# Junior Developer Implementation Guide: Issue #345

## Overview
**Issue**: Regression Loss Functions Unit Tests
**Goal**: Create comprehensive unit tests for regression-focused FitnessCalculators
**Difficulty**: Beginner-Friendly
**Estimated Time**: 4-6 hours

## What You'll Be Testing

You'll create unit tests for **8 regression loss function fitness calculators**:

1. **MeanSquaredErrorFitnessCalculator** - Most common loss for regression
2. **MeanAbsoluteErrorFitnessCalculator** - Less sensitive to outliers
3. **HuberLossFitnessCalculator** - Hybrid of MSE and MAE
4. **LogCoshLossFitnessCalculator** - Smooth approximation of MAE
5. **ModifiedHuberLossFitnessCalculator** - Robust variant of Huber Loss
6. **ElasticNetLossFitnessCalculator** - Regularization-based loss
7. **ExponentialLossFitnessCalculator** - Exponentially weighted loss
8. **AdjustedRSquaredFitnessCalculator** - Statistical goodness-of-fit metric

## Understanding the Codebase

### Key Files to Review

**Interface:**
```
C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IFitnessCalculator.cs
```

**Base Class:**
```
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\FitnessCalculatorBase.cs
```

**Implementations to Test:**
```
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\MeanSquaredErrorFitnessCalculator.cs
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\MeanAbsoluteErrorFitnessCalculator.cs
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\HuberLossFitnessCalculator.cs
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\LogCoshLossFitnessCalculator.cs
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\ModifiedHuberLossFitnessCalculator.cs
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\ElasticNetLossFitnessCalculator.cs
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\ExponentialLossFitnessCalculator.cs
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\AdjustedRSquaredFitnessCalculator.cs
```

### How FitnessCalculators Work

1. **Purpose**: Measure how well a model performs on regression tasks
2. **Input**: DataSetStats containing predicted and actual values
3. **Output**: Numeric fitness score (lower is better for loss functions)
4. **Key Property**: `IsHigherScoreBetter` (false for all loss functions)

### Mathematical Formulas

**Mean Squared Error (MSE):**
```
MSE = (1/n) * sum((predicted - actual)^2)
```

**Mean Absolute Error (MAE):**
```
MAE = (1/n) * sum(|predicted - actual|)
```

**Huber Loss:**
```
For |error| <= delta:
    loss = 0.5 * error^2
For |error| > delta:
    loss = delta * (|error| - 0.5 * delta)
```

**Log-Cosh Loss:**
```
loss = (1/n) * sum(log(cosh(predicted - actual)))
```

**Elastic Net Loss:**
```
loss = MSE + alpha * (l1_ratio * L1 + (1 - l1_ratio) * L2)
where:
    L1 = sum(|weights|)
    L2 = sum(weights^2)
```

## Step-by-Step Implementation Guide

### Step 1: Create Test File Structure

Create file: `C:\Users\cheat\source\repos\AiDotNet\tests\FitnessCalculators\RegressionLossFitnessCalculatorTests.cs`

```csharp
using System;
using AiDotNet.FitnessCalculators;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNet.Tests.FitnessCalculators
{
    public class RegressionLossFitnessCalculatorTests
    {
        // Helper method for floating-point comparison
        private static void AssertClose(double actual, double expected, double tolerance = 1e-6)
        {
            Assert.True(Math.Abs(actual - expected) <= tolerance,
                $"Expected {expected}, but got {actual}. Difference: {Math.Abs(actual - expected)}");
        }

        // Helper method to create test data
        private DataSetStats<double, double, double> CreateTestDataSet(
            double[] predicted,
            double[] actual)
        {
            // Implementation will use ConversionsHelper to create proper data structures
            // This is a simplified example
            return new DataSetStats<double, double, double>
            {
                Predicted = predicted,
                Actual = actual,
                ErrorStats = CalculateErrorStats(predicted, actual)
            };
        }

        private ErrorStats<double> CalculateErrorStats(double[] predicted, double[] actual)
        {
            int n = predicted.Length;
            double mse = 0;
            double mae = 0;

            for (int i = 0; i < n; i++)
            {
                double error = predicted[i] - actual[i];
                mse += error * error;
                mae += Math.Abs(error);
            }

            mse /= n;
            mae /= n;

            return new ErrorStats<double>
            {
                MSE = mse,
                MAE = mae,
                RMSE = Math.Sqrt(mse)
            };
        }

        // Tests will go here
    }
}
```

### Step 2: Test Mean Squared Error (MSE)

**Known Test Cases:**

```csharp
[Fact]
public void MeanSquaredError_PerfectPredictions_ReturnsZero()
{
    // Arrange
    var predicted = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
    var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new MeanSquaredErrorFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    AssertClose(score, 0.0);
    Assert.False(calculator.IsHigherScoreBetter);
}

[Fact]
public void MeanSquaredError_UniformError_ReturnsCorrectValue()
{
    // Arrange
    // All predictions are off by 1.0
    var predicted = new[] { 2.0, 3.0, 4.0, 5.0, 6.0 };
    var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new MeanSquaredErrorFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // MSE = (1^2 + 1^2 + 1^2 + 1^2 + 1^2) / 5 = 5/5 = 1.0
    AssertClose(score, 1.0);
}

[Fact]
public void MeanSquaredError_MixedErrors_ReturnsCorrectValue()
{
    // Arrange
    var predicted = new[] { 3.0, 8.0, 2.0, 6.0 };
    var actual = new[] { 2.0, 5.0, 1.0, 4.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new MeanSquaredErrorFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // Errors: [1, 3, 1, 2]
    // MSE = (1 + 9 + 1 + 4) / 4 = 15/4 = 3.75
    AssertClose(score, 3.75);
}

[Fact]
public void MeanSquaredError_NegativeValues_HandlesCorrectly()
{
    // Arrange
    var predicted = new[] { -1.0, -2.0, -3.0 };
    var actual = new[] { -2.0, -3.0, -4.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new MeanSquaredErrorFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // Errors: [1, 1, 1]
    // MSE = (1 + 1 + 1) / 3 = 1.0
    AssertClose(score, 1.0);
}
```

### Step 3: Test Mean Absolute Error (MAE)

```csharp
[Fact]
public void MeanAbsoluteError_PerfectPredictions_ReturnsZero()
{
    var predicted = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
    var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new MeanAbsoluteErrorFitnessCalculator<double, double, double>();

    var score = calculator.CalculateFitnessScore(dataSet);

    AssertClose(score, 0.0);
}

[Fact]
public void MeanAbsoluteError_MixedErrors_ReturnsCorrectValue()
{
    var predicted = new[] { 3.0, 8.0, 2.0, 6.0 };
    var actual = new[] { 2.0, 5.0, 1.0, 4.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new MeanAbsoluteErrorFitnessCalculator<double, double, double>();

    var score = calculator.CalculateFitnessScore(dataSet);

    // MAE = (|1| + |3| + |1| + |2|) / 4 = 7/4 = 1.75
    AssertClose(score, 1.75);
}

[Fact]
public void MeanAbsoluteError_WithOutlier_LessSensitiveThanMSE()
{
    var predicted = new[] { 1.0, 2.0, 3.0, 100.0 };
    var actual = new[] { 1.0, 2.0, 3.0, 4.0 };
    var dataSet = CreateTestDataSet(predicted, actual);

    var maeCalculator = new MeanAbsoluteErrorFitnessCalculator<double, double, double>();
    var mseCalculator = new MeanSquaredErrorFitnessCalculator<double, double, double>();

    var mae = maeCalculator.CalculateFitnessScore(dataSet);
    var mse = mseCalculator.CalculateFitnessScore(dataSet);

    // MAE = (0 + 0 + 0 + 96) / 4 = 24.0
    // MSE = (0 + 0 + 0 + 9216) / 4 = 2304.0
    AssertClose(mae, 24.0);
    AssertClose(mse, 2304.0);

    // MAE should be much less than MSE due to outlier
    Assert.True(mae < mse);
}
```

### Step 4: Test Huber Loss

```csharp
[Fact]
public void HuberLoss_SmallErrors_BehavesLikeMSE()
{
    // Arrange - errors smaller than delta (1.0)
    var predicted = new[] { 1.5, 2.5, 3.5 };
    var actual = new[] { 1.0, 2.0, 3.0 };
    var delta = 1.0;
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new HuberLossFitnessCalculator<double, double, double>(delta);

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // All errors are 0.5, which is < delta
    // Huber loss for small errors: 0.5 * error^2
    // Expected: 0.5 * (0.25 + 0.25 + 0.25) / 3 = 0.125
    AssertClose(score, 0.125);
}

[Fact]
public void HuberLoss_LargeErrors_BehavesLikeMAE()
{
    // Arrange - errors larger than delta (1.0)
    var predicted = new[] { 5.0, 10.0 };
    var actual = new[] { 1.0, 2.0 };
    var delta = 1.0;
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new HuberLossFitnessCalculator<double, double, double>(delta);

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // Errors: [4.0, 8.0], both > delta
    // Huber loss for large errors: delta * (|error| - 0.5 * delta)
    // loss1 = 1.0 * (4.0 - 0.5) = 3.5
    // loss2 = 1.0 * (8.0 - 0.5) = 7.5
    // Average: (3.5 + 7.5) / 2 = 5.5
    AssertClose(score, 5.5);
}

[Fact]
public void HuberLoss_DifferentDeltas_ProducesDifferentResults()
{
    var predicted = new[] { 5.0, 10.0 };
    var actual = new[] { 1.0, 2.0 };
    var dataSet = CreateTestDataSet(predicted, actual);

    var calculator1 = new HuberLossFitnessCalculator<double, double, double>(1.0);
    var calculator2 = new HuberLossFitnessCalculator<double, double, double>(2.0);

    var score1 = calculator1.CalculateFitnessScore(dataSet);
    var score2 = calculator2.CalculateFitnessScore(dataSet);

    // Different deltas should produce different scores
    Assert.NotEqual(score1, score2);
}
```

### Step 5: Test Log-Cosh Loss

```csharp
[Fact]
public void LogCoshLoss_PerfectPredictions_ReturnsZero()
{
    var predicted = new[] { 1.0, 2.0, 3.0 };
    var actual = new[] { 1.0, 2.0, 3.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new LogCoshLossFitnessCalculator<double, double, double>();

    var score = calculator.CalculateFitnessScore(dataSet);

    AssertClose(score, 0.0);
}

[Fact]
public void LogCoshLoss_SmallErrors_ApproximatelyQuadratic()
{
    // For small errors, log(cosh(x)) ≈ x^2 / 2
    var predicted = new[] { 0.1, 0.2, 0.3 };
    var actual = new[] { 0.0, 0.0, 0.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new LogCoshLossFitnessCalculator<double, double, double>();

    var score = calculator.CalculateFitnessScore(dataSet);

    // For small x, log(cosh(x)) ≈ x^2/2
    // Expected ≈ (0.01/2 + 0.04/2 + 0.09/2) / 3 = 0.14/6 ≈ 0.0233
    Assert.True(score > 0);
    Assert.True(score < 0.1);
}
```

### Step 6: Test Edge Cases (All Calculators)

```csharp
[Fact]
public void AllCalculators_SingleValue_HandlesCorrectly()
{
    var predicted = new[] { 5.0 };
    var actual = new[] { 3.0 };
    var dataSet = CreateTestDataSet(predicted, actual);

    var calculators = new IFitnessCalculator<double, double, double>[]
    {
        new MeanSquaredErrorFitnessCalculator<double, double, double>(),
        new MeanAbsoluteErrorFitnessCalculator<double, double, double>(),
        new HuberLossFitnessCalculator<double, double, double>(),
        new LogCoshLossFitnessCalculator<double, double, double>()
    };

    foreach (var calculator in calculators)
    {
        var score = calculator.CalculateFitnessScore(dataSet);
        Assert.True(score >= 0, $"{calculator.GetType().Name} should return non-negative score");
    }
}

[Fact]
public void AllCalculators_ZeroValues_HandlesCorrectly()
{
    var predicted = new[] { 0.0, 0.0, 0.0 };
    var actual = new[] { 0.0, 0.0, 0.0 };
    var dataSet = CreateTestDataSet(predicted, actual);

    var calculators = new IFitnessCalculator<double, double, double>[]
    {
        new MeanSquaredErrorFitnessCalculator<double, double, double>(),
        new MeanAbsoluteErrorFitnessCalculator<double, double, double>(),
        new HuberLossFitnessCalculator<double, double, double>(),
        new LogCoshLossFitnessCalculator<double, double, double>()
    };

    foreach (var calculator in calculators)
    {
        var score = calculator.CalculateFitnessScore(dataSet);
        AssertClose(score, 0.0);
    }
}

[Fact]
public void AllCalculators_IsHigherScoreBetter_ReturnsFalse()
{
    var calculators = new IFitnessCalculator<double, double, double>[]
    {
        new MeanSquaredErrorFitnessCalculator<double, double, double>(),
        new MeanAbsoluteErrorFitnessCalculator<double, double, double>(),
        new HuberLossFitnessCalculator<double, double, double>(),
        new LogCoshLossFitnessCalculator<double, double, double>(),
        new ElasticNetLossFitnessCalculator<double, double, double>(),
        new ExponentialLossFitnessCalculator<double, double, double>()
    };

    foreach (var calculator in calculators)
    {
        Assert.False(calculator.IsHigherScoreBetter,
            $"{calculator.GetType().Name} should have IsHigherScoreBetter = false");
    }
}
```

### Step 7: Test IsBetterFitness Method

```csharp
[Fact]
public void IsBetterFitness_LowerScoreIsBetter_ReturnsCorrectComparison()
{
    var calculator = new MeanSquaredErrorFitnessCalculator<double, double, double>();

    // Lower score (1.0) should be better than higher score (2.0)
    Assert.True(calculator.IsBetterFitness(1.0, 2.0));
    Assert.False(calculator.IsBetterFitness(2.0, 1.0));
    Assert.False(calculator.IsBetterFitness(1.0, 1.0)); // Equal scores
}

[Fact]
public void IsBetterFitness_ZeroScore_IsBestPossible()
{
    var calculator = new MeanSquaredErrorFitnessCalculator<double, double, double>();

    // Zero score should be better than any positive score
    Assert.True(calculator.IsBetterFitness(0.0, 1.0));
    Assert.True(calculator.IsBetterFitness(0.0, 0.1));
    Assert.False(calculator.IsBetterFitness(1.0, 0.0));
}
```

## Test Coverage Checklist

For each regression loss calculator, ensure you have tests for:

- [ ] Perfect predictions (score = 0)
- [ ] Uniform errors (all predictions off by same amount)
- [ ] Mixed positive and negative errors
- [ ] Edge cases:
  - [ ] Single value
  - [ ] All zeros
  - [ ] Negative values
  - [ ] Large values (outliers)
- [ ] Property validation:
  - [ ] IsHigherScoreBetter returns false
  - [ ] IsBetterFitness works correctly
- [ ] Parameter variations (for Huber, ElasticNet, etc.):
  - [ ] Default parameters
  - [ ] Custom parameters
  - [ ] Extreme parameter values

## Running Your Tests

```bash
# Run all tests
dotnet test

# Run only FitnessCalculator tests
dotnet test --filter "FullyQualifiedName~RegressionLossFitnessCalculatorTests"

# Run specific test
dotnet test --filter "FullyQualifiedName~MeanSquaredError_PerfectPredictions_ReturnsZero"
```

## Common Mistakes to Avoid

1. **Forgetting to test negative values** - Loss functions should handle negative predictions/actuals
2. **Not testing edge cases** - Single values, all zeros, outliers are important
3. **Incorrect mathematical calculations** - Double-check your expected values
4. **Not testing parameter variations** - Huber, ElasticNet have configurable parameters
5. **Ignoring floating-point precision** - Use tolerance in comparisons (AssertClose)

## Learning Resources

### Mathematical Background
- **MSE vs MAE**: https://www.statisticshowto.com/mean-squared-error/
- **Huber Loss**: https://en.wikipedia.org/wiki/Huber_loss
- **Elastic Net**: https://en.wikipedia.org/wiki/Elastic_net_regularization

### xUnit Testing
- **xUnit Documentation**: https://xunit.net/docs/getting-started/netfx/visual-studio
- **Fact vs Theory**: https://andrewlock.net/creating-parameterised-tests-in-xunit-with-inlinedata-classdata-and-memberdata/

## Validation Criteria

Your implementation will be considered complete when:

1. All 8 regression loss calculators have comprehensive tests
2. Test coverage includes:
   - Perfect predictions
   - Various error patterns
   - Edge cases (zeros, negatives, outliers)
   - Parameter variations
3. All tests pass successfully
4. Code follows C# naming conventions
5. Test names clearly describe what they're testing

## Questions to Consider

1. Why does MSE penalize outliers more heavily than MAE?
2. When would you choose Huber Loss over MSE or MAE?
3. What's the purpose of the delta parameter in Huber Loss?
4. How does ElasticNet Loss differ from simple MSE?
5. Why is IsHigherScoreBetter false for all loss functions?

## Next Steps After Completion

1. Create a pull request with your tests
2. Review code coverage reports
3. Consider adding property-based tests using FsCheck
4. Write similar tests for classification loss functions (Issue #346)

---

**Good luck!** Remember: testing is about thinking through edge cases and validating behavior, not just achieving coverage numbers. Take your time to understand what each loss function does mathematically before writing tests.
