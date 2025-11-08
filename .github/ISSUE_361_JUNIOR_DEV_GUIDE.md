# Issue #361: Junior Developer Implementation Guide
## Implement Tests for Domain Adaptation and Feature Mapping

---

## Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [Understanding Domain Adaptation](#understanding-domain-adaptation)
3. [Understanding Feature Mapping](#understanding-feature-mapping)
4. [File Structure Overview](#file-structure-overview)
5. [Testing Strategy](#testing-strategy)
6. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
7. [Code Examples](#code-examples)
8. [Validation Checklist](#validation-checklist)

---

## Understanding the Problem

### What Are We Solving?

Domain adaptation and feature mapping components in `src/TransferLearning/` currently have **minimal test coverage** (only basic tests exist). We need to create **comprehensive unit tests** to achieve **80%+ coverage**.

### Files Requiring Tests

1. **CORALDomainAdapter.cs** - CORrelation ALignment adapter (297 lines)
2. **MMDDomainAdapter.cs** - Maximum Mean Discrepancy adapter (238 lines)
3. **IDomainAdapter.cs** - Domain adapter interface (83 lines)
4. **LinearFeatureMapper.cs** - Linear feature projection mapper (305 lines)
5. **IFeatureMapper.cs** - Feature mapper interface (84 lines)

### Why This Matters

Domain adaptation and feature mapping are **critical for transfer learning success**:
- **Domain adaptation** aligns distributions between source and target domains
- **Feature mapping** bridges different feature spaces
- Without proper testing, we can't ensure these components work correctly

**Real-World Impact**:
- A broken CORAL adapter could fail to align covariances, causing transfer to fail
- An incorrect MMD implementation could give wrong discrepancy measures
- A faulty feature mapper could map features incorrectly, destroying information

---

## Understanding Domain Adaptation

### What Is Domain Adaptation?

Domain adaptation reduces the **distribution shift** between source and target domains. Even when features are the same, the statistical properties might differ.

**Real-World Analogy**: Imagine you're used to driving in sunny California (source domain), but now need to drive in snowy Alaska (target domain). The car is the same (same features), but the conditions are different (different distributions). Domain adaptation is like getting winter driving training to adapt to the new conditions.

### Key Concepts

#### 1. Distribution Shift

```
Source Domain:  Mean = [10, 20], Variance = [5, 5]
Target Domain:  Mean = [15, 18], Variance = [3, 8]

Problem: Models trained on source might not work well on target
Solution: Domain adaptation aligns these distributions
```

#### 2. CORAL (CORrelation ALignment)

CORAL aligns the **second-order statistics** (covariances) of source and target domains.

**Mathematics**:
```
1. Compute covariance matrices: Cs (source), Ct (target)
2. Compute transformation: T = Cs^(-1/2) * Ct^(1/2)
3. Transform source: X_adapted = (X - mean_s) * T + mean_t
```

**What It Does**:
- Makes the spread (variance) of features similar
- Makes correlations between features similar
- Preserves the mean shift

**Example**:
```csharp
// Source data: Tightly clustered around [5, 5]
// Target data: Spread out around [10, 10]
// CORAL: Stretches source data to match target's spread
// Result: Source data now has similar variance as target
```

#### 3. MMD (Maximum Mean Discrepancy)

MMD measures **how different two distributions are** using kernel methods.

**Mathematics**:
```
MMD^2(X, Y) = E[k(x, x')] + E[k(y, y')] - 2*E[k(x, y)]

Where:
- k(x, y) is a kernel function (usually Gaussian)
- E[...] is expectation (average)
- X is source domain, Y is target domain
```

**What It Does**:
- Computes a single number representing domain difference
- 0 = identical distributions
- Large value = very different distributions
- Uses kernel trick to compare complex patterns

**Example**:
```csharp
// Source: Images taken indoors
// Target: Images taken outdoors
// MMD: Measures how different average brightness, contrast patterns are
// If MMD = 0.05: Domains are similar (transfer should work)
// If MMD = 2.5: Domains are very different (transfer risky)
```

#### 4. Domain Discrepancy

The measure of **how different** source and target domains are:
- **Low discrepancy**: Transfer learning will likely work well
- **High discrepancy**: Transfer learning might struggle

---

## Understanding Feature Mapping

### What Is Feature Mapping?

Feature mapping transforms data from one **feature space** to another. This is needed when source and target have different numbers or types of features.

**Real-World Analogy**: Imagine translating a book from English (100,000 words) to a language with only 50,000 words. Feature mapping finds the best way to express English concepts using the limited vocabulary available.

### Key Concepts

#### 1. Feature Space Mismatch

```
Source Model: Trained on 100 features (e.g., image pixels)
Target Data: Only has 50 features (e.g., lower resolution)

Problem: Can't directly apply source model to target data
Solution: Feature mapper bridges the gap
```

#### 2. Linear Feature Mapping

Uses **matrix multiplication** to transform features:

```
Source Features (N × 100) * Projection Matrix (100 × 50) = Target Features (N × 50)
```

**What It Does**:
- Projects high-dimensional data to low-dimensional space (or vice versa)
- Learns the projection by analyzing both source and target data
- Uses orthogonalization to preserve information

**Example**:
```csharp
// Source: 100 features (detailed medical measurements)
// Target: 50 features (simplified measurements)
// Mapper: Learns which source features are most important
// Result: Combines source features into 50 meaningful target features
```

#### 3. Bidirectional Mapping

Feature mappers support **two directions**:

```csharp
// Forward: Source → Target
MapToTarget(sourceData, targetDimension)
// Maps 100 features down to 50

// Reverse: Target → Source
MapToSource(targetData, sourceDimension)
// Maps 50 features back up to 100
```

#### 4. Mapping Confidence

A score (0 to 1) indicating **mapping quality**:
- **High confidence (0.9)**: Mapping preserves most information
- **Low confidence (0.3)**: Mapping loses significant information

**Calculation**:
```csharp
// Confidence = exp(-reconstruction_error)
// Reconstruction error: How well we can round-trip data
// Source → Target → Source should be close to original
```

---

## File Structure Overview

### Source Files Location
```
src/TransferLearning/
  DomainAdaptation/
    - IDomainAdapter.cs              (83 lines)  - Interface
    - CORALDomainAdapter.cs          (297 lines) - CORAL implementation
    - MMDDomainAdapter.cs            (238 lines) - MMD implementation

  FeatureMapping/
    - IFeatureMapper.cs              (84 lines)  - Interface
    - LinearFeatureMapper.cs         (305 lines) - Linear mapping
```

### Test Files Location
```
tests/UnitTests/TransferLearning/
  - DomainAdapterTests.cs          (EXISTS - needs expansion)
  - FeatureMapperTests.cs          (EXISTS - needs expansion)
  - CORALDomainAdapterTests.cs     (NEW - comprehensive)
  - MMDDomainAdapterTests.cs       (NEW - comprehensive)
  - LinearFeatureMapperTests.cs    (NEW - comprehensive)
```

### Existing Test Coverage

Current test files have **basic tests only**:
- DomainAdapterTests.cs: 4 tests (initialization, basic adaptation)
- FeatureMapperTests.cs: 4 tests (initialization, basic mapping)

**Gaps to fill**:
- No covariance alignment verification
- No discrepancy measurement tests
- No mapping confidence validation
- No edge case coverage
- No mathematical correctness tests

---

## Testing Strategy

### Goal: 80%+ Test Coverage

To achieve comprehensive coverage, we need to test:

### 1. Interface Contract Tests (IDomainAdapter, IFeatureMapper)
- All methods defined in interface
- Return types correct
- Property getters work

### 2. CORAL Domain Adapter Tests
- **Initialization**: Constructor, properties
- **Training**: Transformation matrix computation
- **Adaptation**: Source/target alignment
- **Covariance**: Verify statistical alignment
- **Discrepancy**: Frobenius norm calculation
- **Edge cases**: Empty data, singular matrices

### 3. MMD Domain Adapter Tests
- **Initialization**: Kernel setup, sigma parameter
- **Discrepancy**: MMD calculation correctness
- **Kernel evaluation**: Gaussian kernel usage
- **Adaptation**: Mean shift application
- **Median heuristic**: Sigma estimation
- **Edge cases**: Identical/very different distributions

### 4. Linear Feature Mapper Tests
- **Training**: Projection matrix learning
- **Mapping**: Bidirectional transformation
- **Confidence**: Reconstruction quality
- **Orthogonalization**: Gram-Schmidt correctness
- **Edge cases**: Dimension mismatches

### 5. Mathematical Correctness Tests
- **CORAL**: Verify covariance alignment
- **MMD**: Verify discrepancy properties
- **Mapping**: Verify information preservation

### 6. Integration Tests
- **CORAL + Mapper**: Combined usage
- **MMD + Mapper**: Combined usage
- **Real transfer scenario**: End-to-end

---

## Step-by-Step Implementation Guide

### Phase 1: Expand DomainAdapterTests.cs (2 hours)

#### Step 1.1: Add Comprehensive CORAL Tests

```csharp
// Add to DomainAdapterTests.cs

#region CORAL Advanced Tests

[Fact]
public void CORALDomainAdapter_Train_ComputesTransformationMatrix()
{
    // Arrange
    var adapter = new CORALDomainAdapter<double>();
    var sourceData = CreateDistribution(20, 3, mean: 5.0, variance: 2.0);
    var targetData = CreateDistribution(20, 3, mean: 10.0, variance: 4.0);

    // Act
    adapter.Train(sourceData, targetData);

    // Assert - After training, adaptation should work
    var adapted = adapter.AdaptSource(sourceData, targetData);
    Assert.Equal(sourceData.Rows, adapted.Rows);
    Assert.Equal(sourceData.Columns, adapted.Columns);
}

[Fact]
public void CORALDomainAdapter_AdaptSource_AlignsCovariance()
{
    // Arrange
    var adapter = new CORALDomainAdapter<double>();

    // Create source with low variance
    var sourceData = CreateDistribution(30, 2, mean: 0.0, variance: 1.0);

    // Create target with high variance
    var targetData = CreateDistribution(30, 2, mean: 0.0, variance: 4.0);

    // Act
    adapter.Train(sourceData, targetData);
    var adapted = adapter.AdaptSource(sourceData, targetData);

    // Assert - Adapted data should have variance closer to target
    double adaptedVariance = ComputeVariance(adapted, columnIndex: 0);
    double targetVariance = ComputeVariance(targetData, columnIndex: 0);

    // Adapted variance should be closer to target than source was
    Assert.True(Math.Abs(adaptedVariance - targetVariance) < 2.0);
}

[Fact]
public void CORALDomainAdapter_AdaptTarget_ReverseAdaptation()
{
    // Arrange
    var adapter = new CORALDomainAdapter<double>();
    var sourceData = CreateDistribution(25, 3, mean: 5.0, variance: 1.5);
    var targetData = CreateDistribution(25, 3, mean: 8.0, variance: 3.0);

    // Act
    adapter.Train(sourceData, targetData);
    var adaptedTarget = adapter.AdaptTarget(targetData, sourceData);

    // Assert - Dimensions should match
    Assert.Equal(targetData.Rows, adaptedTarget.Rows);
    Assert.Equal(targetData.Columns, adaptedTarget.Columns);

    // Mean should shift towards source
    double sourceMean = ComputeMean(sourceData, columnIndex: 0);
    double adaptedMean = ComputeMean(adaptedTarget, columnIndex: 0);
    Assert.True(Math.Abs(adaptedMean - sourceMean) < Math.Abs(8.0 - sourceMean));
}

[Fact]
public void CORALDomainAdapter_ComputeDomainDiscrepancy_FrobeniusNorm()
{
    // Arrange
    var adapter = new CORALDomainAdapter<double>();

    // Identical distributions should have zero discrepancy
    var data1 = CreateDistribution(20, 3, mean: 5.0, variance: 2.0);
    var data2 = CreateDistribution(20, 3, mean: 5.0, variance: 2.0);

    // Act
    var discrepancy = adapter.ComputeDomainDiscrepancy(data1, data2);

    // Assert
    Assert.True(discrepancy >= 0.0); // Always non-negative
    Assert.True(discrepancy < 1.0); // Should be small for similar distributions
}

[Fact]
public void CORALDomainAdapter_ComputeDomainDiscrepancy_IncreasesWithDifference()
{
    // Arrange
    var adapter = new CORALDomainAdapter<double>();
    var sourceData = CreateDistribution(20, 3, mean: 0.0, variance: 1.0);

    var similarTarget = CreateDistribution(20, 3, mean: 0.5, variance: 1.2);
    var differentTarget = CreateDistribution(20, 3, mean: 5.0, variance: 5.0);

    // Act
    var discrepancySimilar = adapter.ComputeDomainDiscrepancy(sourceData, similarTarget);
    var discrepancyDifferent = adapter.ComputeDomainDiscrepancy(sourceData, differentTarget);

    // Assert
    Assert.True(discrepancyDifferent > discrepancySimilar);
}

[Fact]
public void CORALDomainAdapter_AdaptSource_WithoutTraining_TrainsAutomatically()
{
    // Arrange
    var adapter = new CORALDomainAdapter<double>();
    var sourceData = CreateDistribution(15, 2, mean: 3.0, variance: 1.0);
    var targetData = CreateDistribution(15, 2, mean: 7.0, variance: 2.0);

    // Act - Call AdaptSource without calling Train first
    var adapted = adapter.AdaptSource(sourceData, targetData);

    // Assert - Should work (trains automatically)
    Assert.NotNull(adapted);
    Assert.Equal(sourceData.Rows, adapted.Rows);
}

#endregion

#region Helper Methods

private Matrix<double> CreateDistribution(int samples, int features, double mean, double variance)
{
    var matrix = new Matrix<double>(samples, features);
    var random = new Random(42);

    for (int i = 0; i < samples; i++)
    {
        for (int j = 0; j < features; j++)
        {
            // Box-Muller transform for normal distribution
            double u1 = random.NextDouble();
            double u2 = random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            matrix[i, j] = mean + Math.Sqrt(variance) * randStdNormal;
        }
    }

    return matrix;
}

private double ComputeMean(Matrix<double> data, int columnIndex)
{
    double sum = 0.0;
    for (int i = 0; i < data.Rows; i++)
        sum += data[i, columnIndex];
    return sum / data.Rows;
}

private double ComputeVariance(Matrix<double> data, int columnIndex)
{
    double mean = ComputeMean(data, columnIndex);
    double sumSquaredDiff = 0.0;

    for (int i = 0; i < data.Rows; i++)
    {
        double diff = data[i, columnIndex] - mean;
        sumSquaredDiff += diff * diff;
    }

    return sumSquaredDiff / data.Rows;
}

#endregion
```

#### Step 1.2: Add Comprehensive MMD Tests

```csharp
// Add to DomainAdapterTests.cs

#region MMD Advanced Tests

[Fact]
public void MMDDomainAdapter_Constructor_DefaultSigma()
{
    // Arrange & Act
    var adapter = new MMDDomainAdapter<double>();

    // Assert
    Assert.Equal("Maximum Mean Discrepancy (MMD)", adapter.AdaptationMethod);
    Assert.False(adapter.RequiresTraining);
}

[Fact]
public void MMDDomainAdapter_Constructor_CustomSigma()
{
    // Arrange & Act
    var adapter = new MMDDomainAdapter<double>(sigma: 2.5);

    // Assert
    Assert.NotNull(adapter);
}

[Fact]
public void MMDDomainAdapter_Train_ComputesMedianHeuristic()
{
    // Arrange
    var adapter = new MMDDomainAdapter<double>();
    var sourceData = CreateDistribution(30, 4, mean: 0.0, variance: 1.0);
    var targetData = CreateDistribution(30, 4, mean: 2.0, variance: 1.5);

    // Act
    adapter.Train(sourceData, targetData);

    // Assert - Training should succeed (updates sigma internally)
    var discrepancy = adapter.ComputeDomainDiscrepancy(sourceData, targetData);
    Assert.True(discrepancy >= 0.0);
}

[Fact]
public void MMDDomainAdapter_ComputeDomainDiscrepancy_IdenticalData_ReturnsZero()
{
    // Arrange
    var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
    var data = CreateDistribution(25, 3, mean: 5.0, variance: 2.0);

    // Act
    var discrepancy = adapter.ComputeDomainDiscrepancy(data, data);

    // Assert
    // Identical data should have zero or near-zero discrepancy
    Assert.True(discrepancy < 0.01);
}

[Fact]
public void MMDDomainAdapter_ComputeDomainDiscrepancy_DifferentData_ReturnsPositive()
{
    // Arrange
    var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
    var sourceData = CreateDistribution(20, 3, mean: 0.0, variance: 1.0);
    var targetData = CreateDistribution(20, 3, mean: 10.0, variance: 3.0);

    // Act
    var discrepancy = adapter.ComputeDomainDiscrepancy(sourceData, targetData);

    // Assert
    Assert.True(discrepancy > 0.1); // Should be significantly positive
}

[Fact]
public void MMDDomainAdapter_ComputeDomainDiscrepancy_IsSymmetric()
{
    // Arrange
    var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
    var data1 = CreateDistribution(15, 2, mean: 3.0, variance: 1.0);
    var data2 = CreateDistribution(15, 2, mean: 6.0, variance: 2.0);

    // Act
    var discrepancy12 = adapter.ComputeDomainDiscrepancy(data1, data2);
    var discrepancy21 = adapter.ComputeDomainDiscrepancy(data2, data1);

    // Assert
    // MMD should be symmetric: MMD(X, Y) = MMD(Y, X)
    Assert.Equal(discrepancy12, discrepancy21, precision: 6);
}

[Fact]
public void MMDDomainAdapter_AdaptSource_ShiftsMean()
{
    // Arrange
    var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
    var sourceData = CreateDistribution(20, 2, mean: 0.0, variance: 1.0);
    var targetData = CreateDistribution(20, 2, mean: 5.0, variance: 1.0);

    // Act
    var adapted = adapter.AdaptSource(sourceData, targetData);

    // Assert
    // Adapted mean should be closer to target mean
    double adaptedMean = ComputeMean(adapted, columnIndex: 0);
    double targetMean = ComputeMean(targetData, columnIndex: 0);

    Assert.True(Math.Abs(adaptedMean - targetMean) < Math.Abs(0.0 - targetMean));
}

[Fact]
public void MMDDomainAdapter_AdaptTarget_ReverseShift()
{
    // Arrange
    var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
    var sourceData = CreateDistribution(20, 2, mean: 0.0, variance: 1.0);
    var targetData = CreateDistribution(20, 2, mean: 5.0, variance: 1.0);

    // Act
    var adapted = adapter.AdaptTarget(targetData, sourceData);

    // Assert
    double adaptedMean = ComputeMean(adapted, columnIndex: 0);
    double sourceMean = ComputeMean(sourceData, columnIndex: 0);

    Assert.True(Math.Abs(adaptedMean - sourceMean) < Math.Abs(5.0 - sourceMean));
}

[Fact]
public void MMDDomainAdapter_DifferentSigma_AffectsDiscrepancy()
{
    // Arrange
    var smallSigma = new MMDDomainAdapter<double>(sigma: 0.1);
    var largeSigma = new MMDDomainAdapter<double>(sigma: 10.0);

    var sourceData = CreateDistribution(20, 3, mean: 0.0, variance: 1.0);
    var targetData = CreateDistribution(20, 3, mean: 2.0, variance: 1.5);

    // Act
    var discrepancySmall = smallSigma.ComputeDomainDiscrepancy(sourceData, targetData);
    var discrepancyLarge = largeSigma.ComputeDomainDiscrepancy(sourceData, targetData);

    // Assert
    // Different sigma values should yield different discrepancies
    Assert.NotEqual(discrepancySmall, discrepancyLarge, precision: 3);
}

#endregion
```

### Phase 2: Expand FeatureMapperTests.cs (2 hours)

#### Step 2.1: Add Comprehensive Mapping Tests

```csharp
// Add to FeatureMapperTests.cs

#region Advanced Mapping Tests

[Fact]
public void LinearFeatureMapper_MapToTarget_CorrectDimensions()
{
    // Arrange
    var mapper = new LinearFeatureMapper<double>();
    var sourceData = CreateTestMatrix(25, 8);
    var targetData = CreateTestMatrix(25, 4);
    mapper.Train(sourceData, targetData);

    // Act
    var testData = CreateTestMatrix(10, 8); // 8 source features
    var mapped = mapper.MapToTarget(testData, targetDimension: 4);

    // Assert
    Assert.Equal(10, mapped.Rows); // Same number of samples
    Assert.Equal(4, mapped.Columns); // Target dimensions
}

[Fact]
public void LinearFeatureMapper_MapToSource_CorrectDimensions()
{
    // Arrange
    var mapper = new LinearFeatureMapper<double>();
    var sourceData = CreateTestMatrix(25, 6);
    var targetData = CreateTestMatrix(25, 3);
    mapper.Train(sourceData, targetData);

    // Act
    var testData = CreateTestMatrix(10, 3); // 3 target features
    var mapped = mapper.MapToSource(testData, sourceDimension: 6);

    // Assert
    Assert.Equal(10, mapped.Rows);
    Assert.Equal(6, mapped.Columns); // Source dimensions
}

[Fact]
public void LinearFeatureMapper_RoundTrip_PreservesInformation()
{
    // Arrange
    var mapper = new LinearFeatureMapper<double>();
    var sourceData = CreateTestMatrix(30, 8);
    var targetData = CreateTestMatrix(30, 8); // Same dimensions for fair test
    mapper.Train(sourceData, targetData);

    var originalData = CreateTestMatrix(5, 8);

    // Act - Round trip: Source → Target → Source
    var toTarget = mapper.MapToTarget(originalData, targetDimension: 8);
    var backToSource = mapper.MapToSource(toTarget, sourceDimension: 8);

    // Assert - Should be close to original (some loss expected)
    double error = ComputeReconstructionError(originalData, backToSource);
    Assert.True(error < 1.0); // Reasonable reconstruction
}

[Fact]
public void LinearFeatureMapper_GetMappingConfidence_InRange()
{
    // Arrange
    var mapper = new LinearFeatureMapper<double>();
    var sourceData = CreateTestMatrix(20, 10);
    var targetData = CreateTestMatrix(20, 5);

    // Act
    mapper.Train(sourceData, targetData);
    var confidence = mapper.GetMappingConfidence();

    // Assert
    Assert.True(confidence >= 0.0 && confidence <= 1.0);
}

[Fact]
public void LinearFeatureMapper_HigherDimensionPreservation_BetterConfidence()
{
    // Arrange
    var sourceData = CreateTestMatrix(30, 10);

    // Mapper 1: 10 → 8 (small reduction)
    var mapper1 = new LinearFeatureMapper<double>();
    var targetData1 = CreateTestMatrix(30, 8);
    mapper1.Train(sourceData, targetData1);

    // Mapper 2: 10 → 3 (large reduction)
    var mapper2 = new LinearFeatureMapper<double>();
    var targetData2 = CreateTestMatrix(30, 3);
    mapper2.Train(sourceData, targetData2);

    // Act
    var confidence1 = mapper1.GetMappingConfidence();
    var confidence2 = mapper2.GetMappingConfidence();

    // Assert - Smaller reduction should have better confidence
    Assert.True(confidence1 >= confidence2);
}

[Fact]
public void LinearFeatureMapper_MapToTarget_BeforeTraining_ThrowsException()
{
    // Arrange
    var mapper = new LinearFeatureMapper<double>();
    var data = CreateTestMatrix(10, 5);

    // Act & Assert
    var exception = Assert.Throws<InvalidOperationException>(() =>
        mapper.MapToTarget(data, targetDimension: 3));

    Assert.Contains("trained", exception.Message, StringComparison.OrdinalIgnoreCase);
}

[Fact]
public void LinearFeatureMapper_MapToSource_BeforeTraining_ThrowsException()
{
    // Arrange
    var mapper = new LinearFeatureMapper<double>();
    var data = CreateTestMatrix(10, 3);

    // Act & Assert
    var exception = Assert.Throws<InvalidOperationException>(() =>
        mapper.MapToSource(data, sourceDimension: 5));

    Assert.Contains("trained", exception.Message, StringComparison.OrdinalIgnoreCase);
}

[Fact]
public void LinearFeatureMapper_Train_MultipleFeatureSpaces()
{
    // Arrange
    var mapper = new LinearFeatureMapper<double>();

    // Test various dimension combinations
    var testCases = new[]
    {
        (source: 10, target: 5),
        (source: 20, target: 10),
        (source: 5, target: 10),  // Upsampling
        (source: 15, target: 15)  // Same dimensions
    };

    foreach (var (source, target) in testCases)
    {
        // Act
        var sourceData = CreateTestMatrix(25, source);
        var targetData = CreateTestMatrix(25, target);
        mapper.Train(sourceData, targetData);

        // Assert
        Assert.True(mapper.IsTrained);
        Assert.True(mapper.GetMappingConfidence() >= 0.0);
    }
}

[Fact]
public void LinearFeatureMapper_PreservesDataStatistics()
{
    // Arrange
    var mapper = new LinearFeatureMapper<double>();
    var sourceData = CreateTestMatrix(40, 8, baseValue: 5.0);
    var targetData = CreateTestMatrix(40, 4, baseValue: 3.0);
    mapper.Train(sourceData, targetData);

    var testData = CreateTestMatrix(10, 8, baseValue: 5.0);

    // Act
    var mapped = mapper.MapToTarget(testData, targetDimension: 4);

    // Assert - Mapped data should have reasonable statistics
    double mappedMean = ComputeMean(mapped, columnIndex: 0);
    Assert.True(Math.Abs(mappedMean) < 100.0); // No explosion
}

#endregion

#region Helper Methods

private Matrix<double> CreateTestMatrix(int rows, int cols, double baseValue = 0.0)
{
    var matrix = new Matrix<double>(rows, cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            matrix[i, j] = baseValue + i * 0.1 + j * 0.01;
        }
    }
    return matrix;
}

private double ComputeReconstructionError(Matrix<double> original, Matrix<double> reconstructed)
{
    double totalError = 0.0;
    int count = 0;

    for (int i = 0; i < original.Rows; i++)
    {
        for (int j = 0; j < original.Columns; j++)
        {
            double diff = original[i, j] - reconstructed[i, j];
            totalError += diff * diff;
            count++;
        }
    }

    return Math.Sqrt(totalError / count);
}

private double ComputeMean(Matrix<double> data, int columnIndex)
{
    double sum = 0.0;
    for (int i = 0; i < data.Rows; i++)
        sum += data[i, columnIndex];
    return sum / data.Rows;
}

#endregion
```

### Phase 3: Create Dedicated Test Files (2 hours)

#### Step 3.1: Create CORALDomainAdapterTests.cs

```csharp
// File: CORALDomainAdapterTests.cs
using AiDotNet.TransferLearning.DomainAdaptation;
using AiDotNet.LinearAlgebra;
using Xunit;
using System;

namespace AiDotNet.Tests.UnitTests.TransferLearning;

/// <summary>
/// Comprehensive tests for CORAL domain adaptation.
/// </summary>
public class CORALDomainAdapterTests
{
    #region Initialization Tests

    [Fact]
    public void Constructor_InitializesCorrectly()
    {
        // Arrange & Act
        var adapter = new CORALDomainAdapter<double>();

        // Assert
        Assert.Equal("CORAL (CORrelation ALignment)", adapter.AdaptationMethod);
        Assert.True(adapter.RequiresTraining);
    }

    #endregion

    #region Training Tests

    [Fact]
    public void Train_ComputesCovarianceMatrices()
    {
        // Arrange
        var adapter = new CORALDomainAdapter<double>();
        var sourceData = CreateNormalDistribution(30, 4, mean: 0.0, stdDev: 1.0);
        var targetData = CreateNormalDistribution(30, 4, mean: 5.0, stdDev: 2.0);

        // Act
        adapter.Train(sourceData, targetData);

        // Assert - Training should enable adaptation
        var adapted = adapter.AdaptSource(sourceData, targetData);
        Assert.NotNull(adapted);
    }

    [Fact]
    public void Train_WithSingularMatrix_HandlesGracefully()
{
        // Arrange
        var adapter = new CORALDomainAdapter<double>();

        // Create rank-deficient data (all columns identical)
        var sourceData = new Matrix<double>(10, 3);
        for (int i = 0; i < 10; i++)
        {
            sourceData[i, 0] = i * 1.0;
            sourceData[i, 1] = i * 1.0; // Same as column 0
            sourceData[i, 2] = i * 1.0; // Same as column 0
        }
        var targetData = CreateNormalDistribution(10, 3, mean: 5.0, stdDev: 1.0);

        // Act & Assert - Should handle gracefully (regularization prevents singularity)
        adapter.Train(sourceData, targetData);
        var adapted = adapter.AdaptSource(sourceData, targetData);
        Assert.NotNull(adapted);
    }

    #endregion

    #region Covariance Alignment Tests

    [Fact]
    public void AdaptSource_AlignsVariance()
    {
        // Arrange
        var adapter = new CORALDomainAdapter<double>();

        // Source: Low variance
        var sourceData = CreateNormalDistribution(50, 2, mean: 0.0, stdDev: 1.0);

        // Target: High variance
        var targetData = CreateNormalDistribution(50, 2, mean: 0.0, stdDev: 3.0);

        // Act
        adapter.Train(sourceData, targetData);
        var adapted = adapter.AdaptSource(sourceData, targetData);

        // Assert
        double sourceVar = ComputeVariance(sourceData, 0);
        double targetVar = ComputeVariance(targetData, 0);
        double adaptedVar = ComputeVariance(adapted, 0);

        // Adapted variance should be closer to target than source was
        double sourceDiff = Math.Abs(sourceVar - targetVar);
        double adaptedDiff = Math.Abs(adaptedVar - targetVar);

        Assert.True(adaptedDiff < sourceDiff * 1.5); // Allow some tolerance
    }

    [Fact]
    public void AdaptSource_PreservesMeanShift()
    {
        // Arrange
        var adapter = new CORALDomainAdapter<double>();
        var sourceData = CreateNormalDistribution(40, 2, mean: 5.0, stdDev: 1.0);
        var targetData = CreateNormalDistribution(40, 2, mean: 10.0, stdDev: 1.5);

        // Act
        adapter.Train(sourceData, targetData);
        var adapted = adapter.AdaptSource(sourceData, targetData);

        // Assert - Mean should shift towards target
        double sourceMean = ComputeMean(sourceData, 0);
        double targetMean = ComputeMean(targetData, 0);
        double adaptedMean = ComputeMean(adapted, 0);

        // Adapted mean should be closer to target mean
        Assert.True(Math.Abs(adaptedMean - targetMean) < Math.Abs(sourceMean - targetMean));
    }

    #endregion

    #region Discrepancy Tests

    [Fact]
    public void ComputeDomainDiscrepancy_IdenticalDistributions_ReturnsSmallValue()
    {
        // Arrange
        var adapter = new CORALDomainAdapter<double>();
        var data1 = CreateNormalDistribution(30, 3, mean: 5.0, stdDev: 2.0);
        var data2 = CreateNormalDistribution(30, 3, mean: 5.0, stdDev: 2.0);

        // Act
        var discrepancy = adapter.ComputeDomainDiscrepancy(data1, data2);

        // Assert
        Assert.True(discrepancy < 2.0); // Small for similar distributions
    }

    [Fact]
    public void ComputeDomainDiscrepancy_IncreaseWithDifference()
    {
        // Arrange
        var adapter = new CORALDomainAdapter<double>();
        var sourceData = CreateNormalDistribution(30, 3, mean: 0.0, stdDev: 1.0);

        var similarTarget = CreateNormalDistribution(30, 3, mean: 1.0, stdDev: 1.2);
        var differentTarget = CreateNormalDistribution(30, 3, mean: 10.0, stdDev: 5.0);

        // Act
        var similarDiscrepancy = adapter.ComputeDomainDiscrepancy(sourceData, similarTarget);
        var differentDiscrepancy = adapter.ComputeDomainDiscrepancy(sourceData, differentTarget);

        // Assert
        Assert.True(differentDiscrepancy > similarDiscrepancy);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void AdaptSource_EmptyData_HandlesGracefully()
    {
        // Arrange
        var adapter = new CORALDomainAdapter<double>();
        var emptyData = new Matrix<double>(0, 3);
        var targetData = CreateNormalDistribution(20, 3, mean: 5.0, stdDev: 1.0);

        // Act & Assert
        var exception = Assert.ThrowsAny<Exception>(() =>
            adapter.AdaptSource(emptyData, targetData));

        Assert.NotNull(exception);
    }

    [Fact]
    public void AdaptSource_MismatchedDimensions_ThrowsException()
    {
        // Arrange
        var adapter = new CORALDomainAdapter<double>();
        var sourceData = CreateNormalDistribution(20, 3, mean: 0.0, stdDev: 1.0);
        var targetData = CreateNormalDistribution(20, 5, mean: 0.0, stdDev: 1.0); // Different dimensions

        // Act & Assert
        var exception = Assert.ThrowsAny<Exception>(() =>
            adapter.Train(sourceData, targetData));

        Assert.NotNull(exception);
    }

    #endregion

    #region Helper Methods

    private Matrix<double> CreateNormalDistribution(int samples, int features, double mean, double stdDev)
    {
        var matrix = new Matrix<double>(samples, features);
        var random = new Random(42);

        for (int i = 0; i < samples; i++)
        {
            for (int j = 0; j < features; j++)
            {
                double u1 = random.NextDouble();
                double u2 = random.NextDouble();
                double randNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                matrix[i, j] = mean + stdDev * randNormal;
            }
        }

        return matrix;
    }

    private double ComputeMean(Matrix<double> data, int column)
    {
        double sum = 0.0;
        for (int i = 0; i < data.Rows; i++)
            sum += data[i, column];
        return sum / data.Rows;
    }

    private double ComputeVariance(Matrix<double> data, int column)
    {
        double mean = ComputeMean(data, column);
        double sumSq = 0.0;
        for (int i = 0; i < data.Rows; i++)
        {
            double diff = data[i, column] - mean;
            sumSq += diff * diff;
        }
        return sumSq / data.Rows;
    }

    #endregion
}
```

#### Step 3.2: Create MMDDomainAdapterTests.cs

```csharp
// File: MMDDomainAdapterTests.cs
using AiDotNet.TransferLearning.DomainAdaptation;
using AiDotNet.LinearAlgebra;
using Xunit;
using System;

namespace AiDotNet.Tests.UnitTests.TransferLearning;

/// <summary>
/// Comprehensive tests for MMD domain adaptation.
/// </summary>
public class MMDDomainAdapterTests
{
    #region Initialization Tests

    [Fact]
    public void Constructor_DefaultSigma_Initializes()
    {
        // Arrange & Act
        var adapter = new MMDDomainAdapter<double>();

        // Assert
        Assert.Equal("Maximum Mean Discrepancy (MMD)", adapter.AdaptationMethod);
        Assert.False(adapter.RequiresTraining); // MMD is non-parametric
    }

    [Fact]
    public void Constructor_CustomSigma_Initializes()
    {
        // Arrange & Act
        var adapter = new MMDDomainAdapter<double>(sigma: 2.5);

        // Assert
        Assert.NotNull(adapter);
    }

    #endregion

    #region MMD Computation Tests

    [Fact]
    public void ComputeDomainDiscrepancy_IdenticalData_ReturnsZero()
    {
        // Arrange
        var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
        var data = CreateUniformDistribution(20, 3, min: 0.0, max: 10.0);

        // Act
        var discrepancy = adapter.ComputeDomainDiscrepancy(data, data);

        // Assert
        Assert.True(discrepancy < 0.001); // Near zero for identical data
    }

    [Fact]
    public void ComputeDomainDiscrepancy_DifferentData_ReturnsPositive()
    {
        // Arrange
        var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
        var data1 = CreateUniformDistribution(20, 3, min: 0.0, max: 5.0);
        var data2 = CreateUniformDistribution(20, 3, min: 10.0, max: 15.0);

        // Act
        var discrepancy = adapter.ComputeDomainDiscrepancy(data1, data2);

        // Assert
        Assert.True(discrepancy > 0.1); // Significantly positive
    }

    [Fact]
    public void ComputeDomainDiscrepancy_IsSymmetric()
    {
        // Arrange
        var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
        var data1 = CreateUniformDistribution(15, 2, min: 0.0, max: 5.0);
        var data2 = CreateUniformDistribution(15, 2, min: 5.0, max: 10.0);

        // Act
        var discrepancy12 = adapter.ComputeDomainDiscrepancy(data1, data2);
        var discrepancy21 = adapter.ComputeDomainDiscrepancy(data2, data1);

        // Assert
        Assert.Equal(discrepancy12, discrepancy21, precision: 6);
    }

    [Fact]
    public void ComputeDomainDiscrepancy_AlwaysNonNegative()
    {
        // Arrange
        var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
        var random = new Random(42);

        // Test multiple random pairs
        for (int i = 0; i < 10; i++)
        {
            var data1 = CreateRandomDistribution(20, 3, random);
            var data2 = CreateRandomDistribution(20, 3, random);

            // Act
            var discrepancy = adapter.ComputeDomainDiscrepancy(data1, data2);

            // Assert
            Assert.True(discrepancy >= 0.0);
        }
    }

    #endregion

    #region Adaptation Tests

    [Fact]
    public void AdaptSource_ShiftsMeanTowardTarget()
    {
        // Arrange
        var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
        var sourceData = CreateUniformDistribution(30, 2, min: 0.0, max: 5.0); // Mean ≈ 2.5
        var targetData = CreateUniformDistribution(30, 2, min: 10.0, max: 15.0); // Mean ≈ 12.5

        // Act
        var adapted = adapter.AdaptSource(sourceData, targetData);

        // Assert
        double sourceMean = ComputeMean(sourceData, 0);
        double targetMean = ComputeMean(targetData, 0);
        double adaptedMean = ComputeMean(adapted, 0);

        // Adapted should be closer to target than source was
        Assert.True(Math.Abs(adaptedMean - targetMean) < Math.Abs(sourceMean - targetMean));
    }

    [Fact]
    public void AdaptTarget_ShiftsMeanTowardSource()
    {
        // Arrange
        var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
        var sourceData = CreateUniformDistribution(30, 2, min: 0.0, max: 5.0);
        var targetData = CreateUniformDistribution(30, 2, min: 10.0, max: 15.0);

        // Act
        var adapted = adapter.AdaptTarget(targetData, sourceData);

        // Assert
        double sourceMean = ComputeMean(sourceData, 0);
        double targetMean = ComputeMean(targetData, 0);
        double adaptedMean = ComputeMean(adapted, 0);

        Assert.True(Math.Abs(adaptedMean - sourceMean) < Math.Abs(targetMean - sourceMean));
    }

    #endregion

    #region Sigma Parameter Tests

    [Fact]
    public void Train_ComputesMedianHeuristic()
    {
        // Arrange
        var adapter = new MMDDomainAdapter<double>(); // No sigma specified
        var data1 = CreateUniformDistribution(30, 3, min: 0.0, max: 10.0);
        var data2 = CreateUniformDistribution(30, 3, min: 5.0, max: 15.0);

        // Act
        adapter.Train(data1, data2);

        // Assert - Should compute and use median heuristic
        var discrepancy = adapter.ComputeDomainDiscrepancy(data1, data2);
        Assert.True(discrepancy >= 0.0);
    }

    [Fact]
    public void DifferentSigma_AffectsDiscrepancy()
    {
        // Arrange
        var smallSigma = new MMDDomainAdapter<double>(sigma: 0.1);
        var largeSigma = new MMDDomainAdapter<double>(sigma: 100.0);
        var data1 = CreateUniformDistribution(20, 3, min: 0.0, max: 5.0);
        var data2 = CreateUniformDistribution(20, 3, min: 3.0, max: 8.0);

        // Act
        var discrepancySmall = smallSigma.ComputeDomainDiscrepancy(data1, data2);
        var discrepancyLarge = largeSigma.ComputeDomainDiscrepancy(data1, data2);

        // Assert
        Assert.NotEqual(discrepancySmall, discrepancyLarge);
    }

    #endregion

    #region Helper Methods

    private Matrix<double> CreateUniformDistribution(int samples, int features, double min, double max)
    {
        var matrix = new Matrix<double>(samples, features);
        var random = new Random(42);

        for (int i = 0; i < samples; i++)
        {
            for (int j = 0; j < features; j++)
            {
                matrix[i, j] = min + random.NextDouble() * (max - min);
            }
        }

        return matrix;
    }

    private Matrix<double> CreateRandomDistribution(int samples, int features, Random random)
    {
        var matrix = new Matrix<double>(samples, features);

        for (int i = 0; i < samples; i++)
        {
            for (int j = 0; j < features; j++)
            {
                matrix[i, j] = random.NextDouble() * 20.0 - 10.0;
            }
        }

        return matrix;
    }

    private double ComputeMean(Matrix<double> data, int column)
    {
        double sum = 0.0;
        for (int i = 0; i < data.Rows; i++)
            sum += data[i, column];
        return sum / data.Rows;
    }

    #endregion
}
```

### Phase 4: Integration Tests (1 hour)

```csharp
// Add to a new file: DomainAdaptationIntegrationTests.cs
using AiDotNet.TransferLearning.DomainAdaptation;
using AiDotNet.TransferLearning.FeatureMapping;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TransferLearning;

public class DomainAdaptationIntegrationTests
{
    [Fact]
    public void Integration_CORAL_WithFeatureMapper()
    {
        // Arrange
        var mapper = new LinearFeatureMapper<double>();
        var adapter = new CORALDomainAdapter<double>();

        // Source: 8 features
        var sourceData = CreateTestMatrix(30, 8, baseValue: 0.0);

        // Target: 4 features (different space)
        var targetData = CreateTestMatrix(30, 4, baseValue: 5.0);

        // Act
        mapper.Train(sourceData, targetData);
        var mappedSource = mapper.MapToTarget(sourceData, targetDimension: 4);

        adapter.Train(mappedSource, targetData);
        var adapted = adapter.AdaptSource(mappedSource, targetData);

        // Assert
        Assert.NotNull(adapted);
        Assert.Equal(30, adapted.Rows);
        Assert.Equal(4, adapted.Columns);
    }

    [Fact]
    public void Integration_MMD_WithFeatureMapper()
    {
        // Arrange
        var mapper = new LinearFeatureMapper<double>();
        var adapter = new MMDDomainAdapter<double>(sigma: 1.0);

        var sourceData = CreateTestMatrix(25, 10, baseValue: 0.0);
        var targetData = CreateTestMatrix(25, 5, baseValue: 3.0);

        // Act
        mapper.Train(sourceData, targetData);
        var mappedSource = mapper.MapToTarget(sourceData, targetDimension: 5);

        var discrepancy = adapter.ComputeDomainDiscrepancy(mappedSource, targetData);
        var adapted = adapter.AdaptSource(mappedSource, targetData);

        // Assert
        Assert.True(discrepancy >= 0.0);
        Assert.NotNull(adapted);
    }

    private Matrix<double> CreateTestMatrix(int rows, int cols, double baseValue)
    {
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = baseValue + i * 0.1 + j * 0.01;
            }
        }
        return matrix;
    }
}
```

---

## Code Examples

### Complete Helper Methods Library

Here's a useful helper class for all test files:

```csharp
// File: TestHelpers/DistributionGenerator.cs
using AiDotNet.LinearAlgebra;
using System;

namespace AiDotNet.Tests.UnitTests.TransferLearning.TestHelpers;

/// <summary>
/// Helper methods for generating test distributions.
/// </summary>
public static class DistributionGenerator
{
    /// <summary>
    /// Creates a matrix with normal (Gaussian) distribution.
    /// </summary>
    public static Matrix<double> CreateNormalDistribution(
        int samples,
        int features,
        double mean,
        double stdDev,
        int seed = 42)
    {
        var matrix = new Matrix<double>(samples, features);
        var random = new Random(seed);

        for (int i = 0; i < samples; i++)
        {
            for (int j = 0; j < features; j++)
            {
                matrix[i, j] = GenerateNormalRandom(random, mean, stdDev);
            }
        }

        return matrix;
    }

    /// <summary>
    /// Creates a matrix with uniform distribution.
    /// </summary>
    public static Matrix<double> CreateUniformDistribution(
        int samples,
        int features,
        double min,
        double max,
        int seed = 42)
    {
        var matrix = new Matrix<double>(samples, features);
        var random = new Random(seed);

        for (int i = 0; i < samples; i++)
        {
            for (int j = 0; j < features; j++)
            {
                matrix[i, j] = min + random.NextDouble() * (max - min);
            }
        }

        return matrix;
    }

    /// <summary>
    /// Generates a single normal (Gaussian) random value using Box-Muller transform.
    /// </summary>
    private static double GenerateNormalRandom(Random random, double mean, double stdDev)
    {
        double u1 = random.NextDouble();
        double u2 = random.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stdDev * randStdNormal;
    }

    /// <summary>
    /// Computes the mean of a matrix column.
    /// </summary>
    public static double ComputeMean(Matrix<double> data, int column)
    {
        double sum = 0.0;
        for (int i = 0; i < data.Rows; i++)
            sum += data[i, column];
        return sum / data.Rows;
    }

    /// <summary>
    /// Computes the variance of a matrix column.
    /// </summary>
    public static double ComputeVariance(Matrix<double> data, int column)
    {
        double mean = ComputeMean(data, column);
        double sumSq = 0.0;

        for (int i = 0; i < data.Rows; i++)
        {
            double diff = data[i, column] - mean;
            sumSq += diff * diff;
        }

        return sumSq / data.Rows;
    }

    /// <summary>
    /// Computes reconstruction error between two matrices.
    /// </summary>
    public static double ComputeReconstructionError(Matrix<double> original, Matrix<double> reconstructed)
    {
        if (original.Rows != reconstructed.Rows || original.Columns != reconstructed.Columns)
            throw new ArgumentException("Matrix dimensions must match");

        double totalError = 0.0;
        int count = original.Rows * original.Columns;

        for (int i = 0; i < original.Rows; i++)
        {
            for (int j = 0; j < original.Columns; j++)
            {
                double diff = original[i, j] - reconstructed[i, j];
                totalError += diff * diff;
            }
        }

        return Math.Sqrt(totalError / count);
    }
}
```

---

## Validation Checklist

### Before Submitting Your PR

- [ ] **All test files created/expanded**
  - DomainAdapterTests.cs (expanded)
  - FeatureMapperTests.cs (expanded)
  - CORALDomainAdapterTests.cs (new)
  - MMDDomainAdapterTests.cs (new)
  - DomainAdaptationIntegrationTests.cs (new)

- [ ] **Code coverage achieved**
  - Run: `dotnet test /p:CollectCoverage=true`
  - Verify: 80%+ coverage for domain adaptation

- [ ] **All tests pass**
  - Run: `dotnet test`
  - Zero failures

- [ ] **Test categories covered**
  - [ ] Interface contract tests
  - [ ] CORAL covariance alignment tests
  - [ ] MMD discrepancy tests
  - [ ] Feature mapper bidirectional tests
  - [ ] Mathematical correctness tests
  - [ ] Edge case tests
  - [ ] Integration tests

- [ ] **Code quality**
  - [ ] No compiler warnings
  - [ ] Following existing test patterns
  - [ ] Proper Arrange-Act-Assert structure
  - [ ] Descriptive test names
  - [ ] Comments for complex scenarios

- [ ] **Mathematical verification**
  - [ ] CORAL alignment verified statistically
  - [ ] MMD symmetry verified
  - [ ] Feature mapping reconstruction error reasonable

### Running Tests Locally

```bash
# Run all transfer learning tests
dotnet test --filter "FullyQualifiedName~TransferLearning"

# Run only domain adapter tests
dotnet test --filter "FullyQualifiedName~DomainAdapter"

# Run only feature mapper tests
dotnet test --filter "FullyQualifiedName~FeatureMapper"

# Run with code coverage
dotnet test /p:CollectCoverage=true /p:CoverletOutputFormat=opencover
```

### Common Issues and Solutions

**Issue**: CORAL tests fail with singular matrix errors
- **Solution**: Add regularization term (1e-5) to diagonal when computing covariances

**Issue**: MMD always returns zero
- **Solution**: Check kernel implementation and sigma parameter

**Issue**: Feature mapper confidence always 1.0
- **Solution**: Verify reconstruction error calculation in mapper

**Issue**: Tests are flaky (sometimes pass, sometimes fail)
- **Solution**: Use fixed random seeds (Random(42)) for reproducibility

---

## Summary

This guide provides comprehensive instructions for testing domain adaptation and feature mapping. Focus on:

1. **Understanding domain adaptation concepts** (CORAL vs MMD)
2. **Testing mathematical correctness** (covariance alignment, discrepancy)
3. **Verifying statistical properties** (variance, mean shifts)
4. **Testing feature mapping** (bidirectional, confidence)
5. **Integration testing** (combining components)

The goal is **80%+ code coverage** with **meaningful tests** that verify mathematical correctness and practical usefulness.

Good luck, and remember: Domain adaptation is about making different worlds compatible!
