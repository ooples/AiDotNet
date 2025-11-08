# Junior Developer Implementation Guide: Issue #346

## Overview
**Issue**: Classification Loss Functions Unit Tests
**Goal**: Create comprehensive unit tests for classification-focused FitnessCalculators
**Difficulty**: Beginner-Friendly
**Estimated Time**: 4-6 hours

## What You'll Be Testing

You'll create unit tests for **5 classification loss function fitness calculators**:

1. **BinaryCrossEntropyLossFitnessCalculator** - Binary classification (yes/no, spam/not spam)
2. **CategoricalCrossEntropyLossFitnessCalculator** - Multi-class classification
3. **CrossEntropyLossFitnessCalculator** - General cross-entropy loss
4. **HingeLossFitnessCalculator** - Support Vector Machine loss
5. **FocalLossFitnessCalculator** - For imbalanced classification problems

## Understanding the Codebase

### Key Files to Review

**Implementations to Test:**
```
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\BinaryCrossEntropyLossFitnessCalculator.cs
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\CategoricalCrossEntropyLossFitnessCalculator.cs
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\CrossEntropyLossFitnessCalculator.cs
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\HingeLossFitnessCalculator.cs
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\FocalLossFitnessCalculator.cs
```

### How Classification Loss Functions Work

1. **Purpose**: Measure how well a model predicts class labels or probabilities
2. **Input**: Predicted probabilities and actual class labels
3. **Output**: Loss score (lower is better)
4. **Key Property**: `IsHigherScoreBetter` is false for all loss functions

### Mathematical Formulas

**Binary Cross-Entropy:**
```
BCE = -1/n * sum(y * log(p) + (1-y) * log(1-p))

where:
    y = actual label (0 or 1)
    p = predicted probability
    n = number of samples
```

**Categorical Cross-Entropy:**
```
CCE = -1/n * sum(sum(y_i * log(p_i)))

where:
    y_i = one-hot encoded actual class
    p_i = predicted probabilities for each class
```

**Hinge Loss:**
```
For binary classification:
    Hinge = max(0, 1 - y * pred)

where:
    y = actual label (-1 or +1)
    pred = predicted score (not probability)
```

**Focal Loss:**
```
FL = -alpha * (1 - p)^gamma * log(p)

where:
    p = predicted probability
    gamma = focusing parameter (default 2.0)
    alpha = weighting factor (default 0.25)
```

## Step-by-Step Implementation Guide

### Step 1: Create Test File Structure

Create file: `C:\Users\cheat\source\repos\AiDotNet\tests\FitnessCalculators\ClassificationLossFitnessCalculatorTests.cs`

```csharp
using System;
using AiDotNet.FitnessCalculators;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNet.Tests.FitnessCalculators
{
    public class ClassificationLossFitnessCalculatorTests
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

### Step 2: Test Binary Cross-Entropy Loss

**Known Test Cases:**

```csharp
[Fact]
public void BinaryCrossEntropy_PerfectPredictions_ReturnsZero()
{
    // Arrange - Perfect predictions
    var predicted = new[] { 1.0, 0.0, 1.0, 0.0 };
    var actual = new[] { 1.0, 0.0, 1.0, 0.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new BinaryCrossEntropyLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // log(1) = 0, so BCE should be 0 for perfect predictions
    AssertClose(score, 0.0);
    Assert.False(calculator.IsHigherScoreBetter);
}

[Fact]
public void BinaryCrossEntropy_FiftyFiftyPredictions_ReturnsHighLoss()
{
    // Arrange - Completely uncertain predictions (0.5)
    var predicted = new[] { 0.5, 0.5, 0.5, 0.5 };
    var actual = new[] { 1.0, 0.0, 1.0, 0.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new BinaryCrossEntropyLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // BCE for p=0.5 should be -log(0.5) = 0.693...
    AssertClose(score, 0.693, tolerance: 0.01);
}

[Fact]
public void BinaryCrossEntropy_ConfidentWrongPredictions_ReturnsVeryHighLoss()
{
    // Arrange - Confidently wrong predictions
    var predicted = new[] { 0.99, 0.01 };
    var actual = new[] { 0.0, 1.0 };  // Opposite of predictions
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new BinaryCrossEntropyLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // Should have very high loss for confident wrong predictions
    Assert.True(score > 3.0);  // -log(0.01) ≈ 4.6
}

[Fact]
public void BinaryCrossEntropy_MixedPredictions_CalculatesCorrectly()
{
    // Arrange
    var predicted = new[] { 0.9, 0.1, 0.8, 0.2 };
    var actual = new[] { 1.0, 0.0, 1.0, 0.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new BinaryCrossEntropyLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // All predictions are confident and correct
    // BCE should be relatively small
    Assert.True(score < 0.2);
    Assert.True(score > 0.0);
}

[Fact]
public void BinaryCrossEntropy_EdgeProbabilities_HandlesCorrectly()
{
    // Arrange - Test numerical stability with extreme probabilities
    // Note: Most implementations clip probabilities to avoid log(0)
    var predicted = new[] { 0.999999, 0.000001 };
    var actual = new[] { 1.0, 0.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new BinaryCrossEntropyLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // Should handle near-zero/near-one probabilities without NaN or infinity
    Assert.True(double.IsFinite(score));
    Assert.True(score >= 0);
}
```

### Step 3: Test Categorical Cross-Entropy Loss

```csharp
[Fact]
public void CategoricalCrossEntropy_PerfectPredictions_ReturnsZero()
{
    // Arrange - One-hot predictions matching actual
    // Predicted: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    // Actual:    [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    var predicted = new[] { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
    var actual = new[] { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new CategoricalCrossEntropyLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    AssertClose(score, 0.0);
}

[Fact]
public void CategoricalCrossEntropy_UniformPredictions_ReturnsMaxEntropy()
{
    // Arrange - Uniform distribution over 3 classes
    var predicted = new[] {
        0.33, 0.33, 0.34,  // Sample 1
        0.33, 0.33, 0.34   // Sample 2
    };
    var actual = new[] {
        1.0, 0.0, 0.0,     // Sample 1 is class 0
        0.0, 1.0, 0.0      // Sample 2 is class 1
    };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new CategoricalCrossEntropyLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // -log(0.33) ≈ 1.1
    AssertClose(score, 1.1, tolerance: 0.1);
}

[Fact]
public void CategoricalCrossEntropy_WrongClass_ReturnsHighLoss()
{
    // Arrange - Confidently predict wrong class
    var predicted = new[] {
        0.01, 0.01, 0.98   // Predicting class 2
    };
    var actual = new[] {
        1.0, 0.0, 0.0      // Actually class 0
    };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new CategoricalCrossEntropyLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // -log(0.01) ≈ 4.6
    Assert.True(score > 4.0);
}
```

### Step 4: Test Hinge Loss

```csharp
[Fact]
public void HingeLoss_PerfectPredictions_ReturnsZero()
{
    // Arrange - Correct predictions with margin > 1
    var predicted = new[] { 2.0, -2.0, 3.0, -3.0 };
    var actual = new[] { 1.0, -1.0, 1.0, -1.0 };  // Using -1/+1 encoding
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new HingeLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // max(0, 1 - y*pred) = max(0, 1 - 2) = 0 for all
    AssertClose(score, 0.0);
}

[Fact]
public void HingeLoss_BarelyCorrect_ReturnsZero()
{
    // Arrange - Predictions exactly at margin boundary
    var predicted = new[] { 1.0, -1.0 };
    var actual = new[] { 1.0, -1.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new HingeLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // max(0, 1 - 1*1) = max(0, 0) = 0
    AssertClose(score, 0.0);
}

[Fact]
public void HingeLoss_WrongPredictions_ReturnsPositiveLoss()
{
    // Arrange - Wrong predictions
    var predicted = new[] { -2.0, 2.0 };
    var actual = new[] { 1.0, -1.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new HingeLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // max(0, 1 - 1*(-2)) = max(0, 3) = 3
    // max(0, 1 - (-1)*2) = max(0, 3) = 3
    // Average: (3 + 3) / 2 = 3.0
    AssertClose(score, 3.0);
}

[Fact]
public void HingeLoss_WeakCorrectPredictions_ReturnsSmallLoss()
{
    // Arrange - Correct but weak predictions (within margin)
    var predicted = new[] { 0.5, -0.5 };
    var actual = new[] { 1.0, -1.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new HingeLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // max(0, 1 - 0.5) = 0.5
    // Average: (0.5 + 0.5) / 2 = 0.5
    AssertClose(score, 0.5);
}
```

### Step 5: Test Focal Loss

```csharp
[Fact]
public void FocalLoss_DefaultParameters_CreatesCorrectly()
{
    // Arrange & Act
    var calculator = new FocalLossFitnessCalculator<double, double, double>();

    // Assert
    Assert.False(calculator.IsHigherScoreBetter);
}

[Fact]
public void FocalLoss_PerfectPredictions_ReturnsZero()
{
    // Arrange
    var predicted = new[] { 0.99, 0.01, 0.99, 0.01 };
    var actual = new[] { 1.0, 0.0, 1.0, 0.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new FocalLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // With gamma=2, (1-0.99)^2 ≈ 0.0001, which makes the loss very small
    Assert.True(score < 0.01);
}

[Fact]
public void FocalLoss_HardExamples_ReceivesMoreWeight()
{
    // Arrange - Hard examples (wrong predictions)
    var predictedHard = new[] { 0.3, 0.7 };  // Uncertain, wrong
    var actualHard = new[] { 1.0, 0.0 };

    // Easy examples (correct predictions)
    var predictedEasy = new[] { 0.9, 0.1 };  // Confident, correct
    var actualEasy = new[] { 1.0, 0.0 };

    var dataSetHard = CreateTestDataSet(predictedHard, actualHard);
    var dataSetEasy = CreateTestDataSet(predictedEasy, actualEasy);
    var calculator = new FocalLossFitnessCalculator<double, double, double>(gamma: 2.0);

    // Act
    var scoreHard = calculator.CalculateFitnessScore(dataSetHard);
    var scoreEasy = calculator.CalculateFitnessScore(dataSetEasy);

    // Assert
    // Hard examples should have higher loss
    Assert.True(scoreHard > scoreEasy);
}

[Fact]
public void FocalLoss_DifferentGammas_ProduceDifferentResults()
{
    // Arrange
    var predicted = new[] { 0.7, 0.3, 0.8, 0.2 };
    var actual = new[] { 1.0, 0.0, 1.0, 0.0 };
    var dataSet = CreateTestDataSet(predicted, actual);

    var calculatorGamma0 = new FocalLossFitnessCalculator<double, double, double>(gamma: 0.0);
    var calculatorGamma2 = new FocalLossFitnessCalculator<double, double, double>(gamma: 2.0);
    var calculatorGamma5 = new FocalLossFitnessCalculator<double, double, double>(gamma: 5.0);

    // Act
    var scoreGamma0 = calculatorGamma0.CalculateFitnessScore(dataSet);
    var scoreGamma2 = calculatorGamma2.CalculateFitnessScore(dataSet);
    var scoreGamma5 = calculatorGamma5.CalculateFitnessScore(dataSet);

    // Assert
    // Gamma=0 should be equivalent to standard cross-entropy (highest loss)
    // Higher gamma should reduce loss for easy examples
    Assert.True(scoreGamma0 > scoreGamma2);
    Assert.True(scoreGamma2 > scoreGamma5);
}

[Fact]
public void FocalLoss_DifferentAlphas_AffectClassBalance()
{
    // Arrange
    var predicted = new[] { 0.8, 0.2 };
    var actual = new[] { 1.0, 0.0 };
    var dataSet = CreateTestDataSet(predicted, actual);

    var calculatorAlpha025 = new FocalLossFitnessCalculator<double, double, double>(alpha: 0.25);
    var calculatorAlpha075 = new FocalLossFitnessCalculator<double, double, double>(alpha: 0.75);

    // Act
    var scoreAlpha025 = calculatorAlpha025.CalculateFitnessScore(dataSet);
    var scoreAlpha075 = calculatorAlpha075.CalculateFitnessScore(dataSet);

    // Assert
    // Different alpha values should produce different scores
    Assert.NotEqual(scoreAlpha025, scoreAlpha075);
}
```

### Step 6: Test Edge Cases (All Classification Calculators)

```csharp
[Fact]
public void AllClassificationCalculators_IsHigherScoreBetter_ReturnsFalse()
{
    // Arrange
    var calculators = new IFitnessCalculator<double, double, double>[]
    {
        new BinaryCrossEntropyLossFitnessCalculator<double, double, double>(),
        new CategoricalCrossEntropyLossFitnessCalculator<double, double, double>(),
        new CrossEntropyLossFitnessCalculator<double, double, double>(),
        new HingeLossFitnessCalculator<double, double, double>(),
        new FocalLossFitnessCalculator<double, double, double>()
    };

    // Act & Assert
    foreach (var calculator in calculators)
    {
        Assert.False(calculator.IsHigherScoreBetter,
            $"{calculator.GetType().Name} should have IsHigherScoreBetter = false");
    }
}

[Fact]
public void BinaryCrossEntropy_InvalidProbabilities_ThrowsOrClips()
{
    // Arrange - Probabilities outside [0, 1]
    var predicted = new[] { 1.5, -0.5 };  // Invalid probabilities
    var actual = new[] { 1.0, 0.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new BinaryCrossEntropyLossFitnessCalculator<double, double, double>();

    // Act & Assert
    // Implementation should either throw exception or clip probabilities
    // This tests robustness of the implementation
    try
    {
        var score = calculator.CalculateFitnessScore(dataSet);
        // If it doesn't throw, check that result is finite
        Assert.True(double.IsFinite(score));
    }
    catch (ArgumentException)
    {
        // Expected behavior if implementation validates input
        Assert.True(true);
    }
}

[Fact]
public void AllClassificationCalculators_SingleSample_HandlesCorrectly()
{
    // Arrange
    var predicted = new[] { 0.8 };
    var actual = new[] { 1.0 };
    var dataSet = CreateTestDataSet(predicted, actual);

    var calculators = new IFitnessCalculator<double, double, double>[]
    {
        new BinaryCrossEntropyLossFitnessCalculator<double, double, double>(),
        new HingeLossFitnessCalculator<double, double, double>(),
        new FocalLossFitnessCalculator<double, double, double>()
    };

    // Act & Assert
    foreach (var calculator in calculators)
    {
        var score = calculator.CalculateFitnessScore(dataSet);
        Assert.True(double.IsFinite(score),
            $"{calculator.GetType().Name} should handle single sample");
        Assert.True(score >= 0,
            $"{calculator.GetType().Name} should return non-negative loss");
    }
}
```

### Step 7: Test IsBetterFitness Method

```csharp
[Fact]
public void IsBetterFitness_LowerLossIsBetter_AllCalculators()
{
    // Arrange
    var calculators = new IFitnessCalculator<double, double, double>[]
    {
        new BinaryCrossEntropyLossFitnessCalculator<double, double, double>(),
        new HingeLossFitnessCalculator<double, double, double>(),
        new FocalLossFitnessCalculator<double, double, double>()
    };

    // Act & Assert
    foreach (var calculator in calculators)
    {
        // Lower loss (0.5) should be better than higher loss (1.5)
        Assert.True(calculator.IsBetterFitness(0.5, 1.5),
            $"{calculator.GetType().Name}: 0.5 should be better than 1.5");

        Assert.False(calculator.IsBetterFitness(1.5, 0.5),
            $"{calculator.GetType().Name}: 1.5 should not be better than 0.5");

        Assert.False(calculator.IsBetterFitness(1.0, 1.0),
            $"{calculator.GetType().Name}: Equal scores should not be better");
    }
}

[Fact]
public void IsBetterFitness_ZeroLossIsBest_AllCalculators()
{
    // Arrange
    var calculators = new IFitnessCalculator<double, double, double>[]
    {
        new BinaryCrossEntropyLossFitnessCalculator<double, double, double>(),
        new HingeLossFitnessCalculator<double, double, double>()
    };

    // Act & Assert
    foreach (var calculator in calculators)
    {
        Assert.True(calculator.IsBetterFitness(0.0, 1.0),
            $"{calculator.GetType().Name}: 0.0 should be better than 1.0");

        Assert.True(calculator.IsBetterFitness(0.0, 0.1),
            $"{calculator.GetType().Name}: 0.0 should be better than 0.1");
    }
}
```

## Test Coverage Checklist

For each classification loss calculator, ensure you have tests for:

- [ ] Perfect predictions (very low loss)
- [ ] Completely wrong predictions (high loss)
- [ ] Uncertain predictions (moderate loss)
- [ ] Edge cases:
  - [ ] Single sample
  - [ ] Extreme probabilities (0, 1)
  - [ ] Invalid probabilities (< 0 or > 1)
  - [ ] Uniform distributions
- [ ] Property validation:
  - [ ] IsHigherScoreBetter returns false
  - [ ] IsBetterFitness works correctly
- [ ] Parameter variations (for Focal Loss):
  - [ ] Different gamma values
  - [ ] Different alpha values
  - [ ] Gamma = 0 (should behave like standard cross-entropy)

## Running Your Tests

```bash
# Run all tests
dotnet test

# Run only classification fitness calculator tests
dotnet test --filter "FullyQualifiedName~ClassificationLossFitnessCalculatorTests"

# Run specific test
dotnet test --filter "FullyQualifiedName~BinaryCrossEntropy_PerfectPredictions"
```

## Common Mistakes to Avoid

1. **Confusing probabilities with class labels** - BCE expects probabilities [0,1], not just 0/1
2. **Not testing numerical stability** - Test edge cases like log(0) or log(1)
3. **Forgetting about multi-class** - Categorical CE works with one-hot encoded vectors
4. **Wrong encoding for Hinge Loss** - Uses -1/+1, not 0/1
5. **Not understanding focal loss parameters** - gamma and alpha dramatically change behavior

## Learning Resources

### Mathematical Background
- **Cross-Entropy**: https://en.wikipedia.org/wiki/Cross_entropy
- **Hinge Loss**: https://en.wikipedia.org/wiki/Hinge_loss
- **Focal Loss Paper**: https://arxiv.org/abs/1708.02002

### Classification Metrics
- **Binary Classification**: https://developers.google.com/machine-learning/crash-course/classification
- **Multi-class Classification**: https://machinelearningmastery.com/types-of-classification-in-machine-learning/

## Validation Criteria

Your implementation will be considered complete when:

1. All 5 classification loss calculators have comprehensive tests
2. Test coverage includes:
   - Perfect, wrong, and uncertain predictions
   - Edge cases (extreme probabilities, single samples)
   - Parameter variations (Focal Loss)
3. All tests pass successfully
4. Test names clearly describe scenarios
5. Mathematical correctness is validated

## Questions to Consider

1. Why does Binary Cross-Entropy heavily penalize confident wrong predictions?
2. How does Categorical Cross-Entropy differ from Binary Cross-Entropy?
3. When would you use Hinge Loss instead of Cross-Entropy?
4. What problem does Focal Loss solve that standard Cross-Entropy doesn't?
5. What happens to Focal Loss when gamma = 0?

## Next Steps After Completion

1. Create a pull request with your tests
2. Review code coverage reports
3. Consider edge cases with very small/large probability values
4. Write tests for specialized loss functions (Issue #376)

---

**Good luck!** Classification loss functions are fundamental to machine learning. Understanding how they behave in different scenarios will make you a better ML engineer.
