# Junior Developer Implementation Guide: Issue #376

## Overview
**Issue**: Specialized Loss Functions Unit Tests
**Goal**: Create comprehensive unit tests for specialized FitnessCalculators used in specific domains
**Difficulty**: Intermediate
**Estimated Time**: 4-6 hours

## What You'll Be Testing

You'll create unit tests for **4 specialized loss function fitness calculators**:

1. **DiceLossFitnessCalculator** - Image segmentation (medical imaging, satellite imagery)
2. **JaccardLossFitnessCalculator** - IoU-based loss for object detection
3. **ContrastiveLossFitnessCalculator** - Similarity learning (face recognition, signature verification)
4. **CosineSimilarityLossFitnessCalculator** - Document similarity, embeddings

## Understanding the Codebase

### Key Files to Review

**Implementations to Test:**
```
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\DiceLossFitnessCalculator.cs
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\JaccardLossFitnessCalculator.cs
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\ContrastiveLossFitnessCalculator.cs
C:\Users\cheat\source\repos\AiDotNet\src\FitnessCalculators\CosineSimilarityLossFitnessCalculator.cs
```

### How Specialized Loss Functions Work

These loss functions are designed for specific use cases:

1. **Dice Loss**: Measures overlap between predicted and actual regions (segmentation)
2. **Jaccard Loss**: Also called IoU (Intersection over Union) loss
3. **Contrastive Loss**: Learns similarity between pairs of items
4. **Cosine Similarity Loss**: Measures angle between vectors (direction, not magnitude)

### Mathematical Formulas

**Dice Loss:**
```
Dice Coefficient = 2 * |A ∩ B| / (|A| + |B|)
Dice Loss = 1 - Dice Coefficient

where:
    A = predicted set
    B = actual set
    |A ∩ B| = intersection (overlap)
    |A| + |B| = sum of sizes
```

**Jaccard Loss (IoU):**
```
Jaccard Index = |A ∩ B| / |A ∪ B|
Jaccard Loss = 1 - Jaccard Index

where:
    |A ∩ B| = intersection
    |A ∪ B| = union
```

**Contrastive Loss:**
```
For similar pairs (y = 1):
    L = distance^2

For dissimilar pairs (y = 0):
    L = max(0, margin - distance)^2

where:
    distance = ||anchor - other||
    margin = minimum separation for dissimilar pairs
```

**Cosine Similarity Loss:**
```
Cosine Similarity = (A · B) / (||A|| * ||B||)
Cosine Distance = 1 - Cosine Similarity

where:
    A · B = dot product
    ||A|| = magnitude of A
```

## Step-by-Step Implementation Guide

### Step 1: Create Test File Structure

Create file: `C:\Users\cheat\source\repos\AiDotNet\tests\FitnessCalculators\SpecializedLossFitnessCalculatorTests.cs`

```csharp
using System;
using AiDotNet.FitnessCalculators;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNet.Tests.FitnessCalculators
{
    public class SpecializedLossFitnessCalculatorTests
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

### Step 2: Test Dice Loss

**Understanding Dice Loss:**
- Perfect overlap: Dice = 1, Loss = 0
- No overlap: Dice = 0, Loss = 1
- Partial overlap: Dice between 0 and 1

```csharp
[Fact]
public void DiceLoss_PerfectOverlap_ReturnsZero()
{
    // Arrange - Perfect segmentation
    var predicted = new[] { 1.0, 1.0, 0.0, 0.0 };
    var actual = new[] { 1.0, 1.0, 0.0, 0.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new DiceLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // Dice = 2*|{1,1}| / (|{1,1}| + |{1,1}|) = 2*2 / (2+2) = 4/4 = 1
    // Loss = 1 - 1 = 0
    AssertClose(score, 0.0);
    Assert.False(calculator.IsHigherScoreBetter);
}

[Fact]
public void DiceLoss_NoOverlap_ReturnsOne()
{
    // Arrange - Completely wrong segmentation
    var predicted = new[] { 1.0, 1.0, 0.0, 0.0 };
    var actual = new[] { 0.0, 0.0, 1.0, 1.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new DiceLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // No intersection: Dice = 0, Loss = 1
    AssertClose(score, 1.0);
}

[Fact]
public void DiceLoss_PartialOverlap_ReturnsCorrectValue()
{
    // Arrange - 50% overlap
    // Predicted: pixels 0,1 are foreground
    // Actual:    pixels 1,2 are foreground
    // Overlap:   pixel 1 only
    var predicted = new[] { 1.0, 1.0, 0.0, 0.0 };
    var actual = new[] { 0.0, 1.0, 1.0, 0.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new DiceLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // Intersection = 1 pixel (index 1)
    // |A| = 2, |B| = 2
    // Dice = 2*1 / (2+2) = 2/4 = 0.5
    // Loss = 1 - 0.5 = 0.5
    AssertClose(score, 0.5);
}

[Fact]
public void DiceLoss_SmallObject_HandlesCorrectly()
{
    // Arrange - Small foreground region (common in medical imaging)
    var predicted = new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0 };
    var actual = new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new DiceLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // Predicted: 2 pixels, Actual: 1 pixel, Overlap: 1 pixel
    // Dice = 2*1 / (2+1) = 2/3 ≈ 0.667
    // Loss = 1 - 0.667 = 0.333
    AssertClose(score, 0.333, tolerance: 0.01);
}

[Fact]
public void DiceLoss_AllZeros_HandlesEdgeCase()
{
    // Arrange - No foreground in either prediction or ground truth
    var predicted = new[] { 0.0, 0.0, 0.0, 0.0 };
    var actual = new[] { 0.0, 0.0, 0.0, 0.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new DiceLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // Edge case: should handle division by zero
    // Common implementations return 0 (perfect match of "no object")
    Assert.True(score >= 0 && score <= 1);
}

[Fact]
public void DiceLoss_ContinuousValues_WorksWithProbabilities()
{
    // Arrange - Soft predictions (probabilities)
    var predicted = new[] { 0.9, 0.8, 0.1, 0.2 };
    var actual = new[] { 1.0, 1.0, 0.0, 0.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new DiceLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // Should handle continuous values (soft Dice)
    Assert.True(score >= 0 && score <= 1);
    Assert.True(score < 0.5); // Good predictions should have low loss
}
```

### Step 3: Test Jaccard Loss (IoU)

```csharp
[Fact]
public void JaccardLoss_PerfectOverlap_ReturnsZero()
{
    // Arrange
    var predicted = new[] { 1.0, 1.0, 0.0, 0.0 };
    var actual = new[] { 1.0, 1.0, 0.0, 0.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new JaccardLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // IoU = |intersection| / |union| = 2/2 = 1
    // Loss = 1 - 1 = 0
    AssertClose(score, 0.0);
}

[Fact]
public void JaccardLoss_NoOverlap_ReturnsOne()
{
    // Arrange
    var predicted = new[] { 1.0, 1.0, 0.0, 0.0 };
    var actual = new[] { 0.0, 0.0, 1.0, 1.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new JaccardLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // IoU = 0/4 = 0, Loss = 1
    AssertClose(score, 1.0);
}

[Fact]
public void JaccardLoss_PartialOverlap_ReturnsCorrectValue()
{
    // Arrange
    var predicted = new[] { 1.0, 1.0, 0.0 };
    var actual = new[] { 0.0, 1.0, 1.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new JaccardLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // Intersection = 1 (index 1)
    // Union = 3 (indices 0, 1, 2)
    // IoU = 1/3 ≈ 0.333
    // Loss = 1 - 0.333 = 0.667
    AssertClose(score, 0.667, tolerance: 0.01);
}

[Fact]
public void JaccardLoss_DifferentFromDiceLoss_OnSameData()
{
    // Arrange
    var predicted = new[] { 1.0, 1.0, 0.0 };
    var actual = new[] { 0.0, 1.0, 1.0 };
    var dataSet = CreateTestDataSet(predicted, actual);

    var diceCalculator = new DiceLossFitnessCalculator<double, double, double>();
    var jaccardCalculator = new JaccardLossFitnessCalculator<double, double, double>();

    // Act
    var diceLoss = diceCalculator.CalculateFitnessScore(dataSet);
    var jaccardLoss = jaccardCalculator.CalculateFitnessScore(dataSet);

    // Assert
    // Dice and Jaccard are related but different metrics
    Assert.NotEqual(diceLoss, jaccardLoss);
    // Jaccard is always <= Dice for same data
    Assert.True(jaccardLoss >= diceLoss);
}
```

### Step 4: Test Contrastive Loss

**Understanding Contrastive Loss:**
- Works with pairs of items
- Similar pairs: penalizes distance
- Dissimilar pairs: penalizes if closer than margin

```csharp
[Fact]
public void ContrastiveLoss_SimilarPairsCloseTogether_ReturnsLowLoss()
{
    // Arrange - Similar items with small distance
    // Pair 1: anchor=[1,2], positive=[1.1,2.1] (very close)
    // Pair 2: anchor=[5,6], positive=[5.1,6.1] (very close)
    var predicted = new[] {
        1.0, 2.0,    // anchor 1
        1.1, 2.1     // positive 1 (similar, close)
    };
    var actual = new[] {
        1.0, 1.0,    // Labels indicating similarity
        1.0, 1.0
    };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new ContrastiveLossFitnessCalculator<double, double, double>(margin: 1.0);

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // Similar pairs close together should have low loss
    Assert.True(score < 0.1);
}

[Fact]
public void ContrastiveLoss_DissimilarPairsFarApart_ReturnsLowLoss()
{
    // Arrange - Dissimilar items with large distance
    var predicted = new[] {
        1.0, 2.0,    // anchor
        10.0, 20.0   // negative (dissimilar, far)
    };
    var actual = new[] {
        0.0, 0.0     // Labels indicating dissimilarity
    };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new ContrastiveLossFitnessCalculator<double, double, double>(margin: 1.0);

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // Dissimilar pairs far apart (beyond margin) should have low loss
    AssertClose(score, 0.0);
}

[Fact]
public void ContrastiveLoss_DissimilarPairsTooClose_ReturnsHighLoss()
{
    // Arrange - Dissimilar items but too close together
    var predicted = new[] {
        1.0, 2.0,    // anchor
        1.2, 2.2     // negative (dissimilar, but close)
    };
    var actual = new[] {
        0.0, 0.0     // Labels indicating dissimilarity
    };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new ContrastiveLossFitnessCalculator<double, double, double>(margin: 2.0);

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // Dissimilar pairs closer than margin should have positive loss
    Assert.True(score > 0);
}

[Fact]
public void ContrastiveLoss_DifferentMargins_ProduceDifferentResults()
{
    // Arrange
    var predicted = new[] { 1.0, 2.0, 2.0, 3.0 };
    var actual = new[] { 0.0, 0.0 };  // Dissimilar pair
    var dataSet = CreateTestDataSet(predicted, actual);

    var calculatorMargin1 = new ContrastiveLossFitnessCalculator<double, double, double>(margin: 1.0);
    var calculatorMargin3 = new ContrastiveLossFitnessCalculator<double, double, double>(margin: 3.0);

    // Act
    var scoreMargin1 = calculatorMargin1.CalculateFitnessScore(dataSet);
    var scoreMargin3 = calculatorMargin3.CalculateFitnessScore(dataSet);

    // Assert
    // Larger margin should result in higher loss for close dissimilar pairs
    Assert.True(scoreMargin3 > scoreMargin1 || scoreMargin3 == 0);
}
```

### Step 5: Test Cosine Similarity Loss

```csharp
[Fact]
public void CosineSimilarity_IdenticalVectors_ReturnsZeroLoss()
{
    // Arrange - Identical vectors (perfect similarity)
    var predicted = new[] { 1.0, 2.0, 3.0 };
    var actual = new[] { 1.0, 2.0, 3.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new CosineSimilarityLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // Cosine similarity = 1.0, Loss = 1 - 1 = 0
    AssertClose(score, 0.0);
}

[Fact]
public void CosineSimilarity_OppositeVectors_ReturnsHighLoss()
{
    // Arrange - Opposite direction vectors
    var predicted = new[] { 1.0, 2.0, 3.0 };
    var actual = new[] { -1.0, -2.0, -3.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new CosineSimilarityLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // Cosine similarity = -1.0, Loss = 1 - (-1) = 2.0
    AssertClose(score, 2.0);
}

[Fact]
public void CosineSimilarity_OrthogonalVectors_ReturnsOneLoss()
{
    // Arrange - Orthogonal vectors (perpendicular)
    var predicted = new[] { 1.0, 0.0 };
    var actual = new[] { 0.0, 1.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new CosineSimilarityLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // Cosine similarity = 0, Loss = 1 - 0 = 1.0
    AssertClose(score, 1.0);
}

[Fact]
public void CosineSimilarity_SameDirection_DifferentMagnitude_ReturnsZero()
{
    // Arrange - Same direction, different magnitude
    var predicted = new[] { 1.0, 2.0, 3.0 };
    var actual = new[] { 2.0, 4.0, 6.0 };  // 2x predicted
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new CosineSimilarityLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // Cosine similarity only cares about direction, not magnitude
    // Should have similarity = 1.0, Loss = 0
    AssertClose(score, 0.0);
}

[Fact]
public void CosineSimilarity_45DegreeAngle_ReturnsCorrectLoss()
{
    // Arrange - Vectors at 45 degrees
    // [1,0] and [1,1] have angle of 45 degrees
    // cos(45°) ≈ 0.707
    var predicted = new[] { 1.0, 0.0 };
    var actual = new[] { 1.0, 1.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new CosineSimilarityLossFitnessCalculator<double, double, double>();

    // Act
    var score = calculator.CalculateFitnessScore(dataSet);

    // Assert
    // Cosine similarity ≈ 0.707
    // Loss ≈ 1 - 0.707 = 0.293
    AssertClose(score, 0.293, tolerance: 0.01);
}

[Fact]
public void CosineSimilarity_ZeroVector_HandlesEdgeCase()
{
    // Arrange - One vector is zero
    var predicted = new[] { 1.0, 2.0, 3.0 };
    var actual = new[] { 0.0, 0.0, 0.0 };
    var dataSet = CreateTestDataSet(predicted, actual);
    var calculator = new CosineSimilarityLossFitnessCalculator<double, double, double>();

    // Act & Assert
    // Should handle division by zero gracefully
    try
    {
        var score = calculator.CalculateFitnessScore(dataSet);
        Assert.True(double.IsFinite(score));
    }
    catch (DivideByZeroException)
    {
        // Acceptable behavior
        Assert.True(true);
    }
}
```

### Step 6: Test Common Properties (All Specialized Calculators)

```csharp
[Fact]
public void AllSpecializedCalculators_IsHigherScoreBetter_ReturnsFalse()
{
    // Arrange
    var calculators = new IFitnessCalculator<double, double, double>[]
    {
        new DiceLossFitnessCalculator<double, double, double>(),
        new JaccardLossFitnessCalculator<double, double, double>(),
        new ContrastiveLossFitnessCalculator<double, double, double>(),
        new CosineSimilarityLossFitnessCalculator<double, double, double>()
    };

    // Act & Assert
    foreach (var calculator in calculators)
    {
        Assert.False(calculator.IsHigherScoreBetter,
            $"{calculator.GetType().Name} should have IsHigherScoreBetter = false");
    }
}

[Fact]
public void AllSpecializedCalculators_IsBetterFitness_LowerIsBetter()
{
    // Arrange
    var calculators = new IFitnessCalculator<double, double, double>[]
    {
        new DiceLossFitnessCalculator<double, double, double>(),
        new JaccardLossFitnessCalculator<double, double, double>(),
        new ContrastiveLossFitnessCalculator<double, double, double>(),
        new CosineSimilarityLossFitnessCalculator<double, double, double>()
    };

    // Act & Assert
    foreach (var calculator in calculators)
    {
        Assert.True(calculator.IsBetterFitness(0.3, 0.7),
            $"{calculator.GetType().Name}: 0.3 should be better than 0.7");

        Assert.False(calculator.IsBetterFitness(0.7, 0.3),
            $"{calculator.GetType().Name}: 0.7 should not be better than 0.3");
    }
}
```

## Test Coverage Checklist

For each specialized loss calculator, ensure you have tests for:

**Dice Loss:**
- [ ] Perfect overlap (loss = 0)
- [ ] No overlap (loss = 1)
- [ ] Partial overlap (loss between 0 and 1)
- [ ] Small objects (imbalanced data)
- [ ] All zeros edge case
- [ ] Continuous values (soft Dice)

**Jaccard Loss:**
- [ ] Perfect overlap
- [ ] No overlap
- [ ] Partial overlap
- [ ] Comparison with Dice Loss
- [ ] All zeros edge case

**Contrastive Loss:**
- [ ] Similar pairs close together
- [ ] Dissimilar pairs far apart
- [ ] Dissimilar pairs too close
- [ ] Different margin values
- [ ] Mixed similar/dissimilar pairs

**Cosine Similarity Loss:**
- [ ] Identical vectors
- [ ] Opposite vectors
- [ ] Orthogonal vectors
- [ ] Same direction, different magnitude
- [ ] Various angles (45°, 90°, 180°)
- [ ] Zero vector edge case

## Running Your Tests

```bash
# Run all tests
dotnet test

# Run only specialized loss tests
dotnet test --filter "FullyQualifiedName~SpecializedLossFitnessCalculatorTests"

# Run specific test
dotnet test --filter "FullyQualifiedName~DiceLoss_PerfectOverlap"
```

## Common Mistakes to Avoid

1. **Confusing Dice and Jaccard** - They're similar but mathematically different
2. **Wrong pair structure for Contrastive Loss** - Need anchor/positive/negative triplets
3. **Forgetting magnitude independence in Cosine Similarity** - [1,2,3] and [2,4,6] are identical
4. **Not testing edge cases** - Empty sets, zero vectors, all same values
5. **Incorrect overlap calculations** - Intersection vs. union vs. sum

## Learning Resources

### Mathematical Background
- **Dice Coefficient**: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
- **Jaccard Index**: https://en.wikipedia.org/wiki/Jaccard_index
- **Contrastive Learning**: https://lilianweng.github.io/posts/2021-05-31-contrastive/
- **Cosine Similarity**: https://en.wikipedia.org/wiki/Cosine_similarity

### Domain Applications
- **Image Segmentation Metrics**: https://www.jeremyjordan.me/evaluating-image-segmentation-models/
- **Similarity Learning**: https://en.wikipedia.org/wiki/Similarity_learning

## Validation Criteria

Your implementation will be considered complete when:

1. All 4 specialized loss calculators have comprehensive tests
2. Test coverage includes:
   - Perfect cases (zero loss)
   - Worst cases (maximum loss)
   - Partial matches
   - Edge cases (zeros, empty sets)
3. All tests pass successfully
4. Mathematical correctness validated with known values
5. Domain-specific scenarios tested (e.g., small objects for Dice)

## Questions to Consider

1. Why is Dice Loss preferred over Jaccard Loss for medical image segmentation?
2. How does Contrastive Loss help in face recognition systems?
3. Why doesn't Cosine Similarity care about vector magnitude?
4. What's the relationship between Dice coefficient and Jaccard index mathematically?
5. When would you use Contrastive Loss vs. Triplet Loss?

## Next Steps After Completion

1. Create a pull request with your tests
2. Consider writing integration tests that combine multiple losses
3. Explore visualization of loss landscapes
4. Write tests for distribution-based loss functions (Issue #377)

---

**Good luck!** These specialized loss functions are used in cutting-edge applications like medical imaging, autonomous vehicles, and face recognition. Understanding them deeply will make you valuable in specialized ML domains.
