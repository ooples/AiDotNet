# Issue #364: Junior Developer Implementation Guide
## Implement Tests for Biorthogonal Wavelets

---

## Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [Understanding Biorthogonal Wavelets](#understanding-biorthogonal-wavelets)
3. [Files Requiring Tests](#files-requiring-tests)
4. [Testing Strategy](#testing-strategy)
5. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
6. [Expected Test Structure](#expected-test-structure)
7. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)

---

## Understanding the Problem

### What Are We Solving?

The biorthogonal wavelet implementations in `src/WaveletFunctions/` currently have **0% test coverage**. We need to achieve **80%+ test coverage** to ensure these wavelets provide perfect reconstruction with symmetry properties.

### Why Testing Matters for Biorthogonal Wavelets

Biorthogonal wavelets are critical for:
- **JPEG2000 image compression** (9/7 wavelet)
- **Medical imaging** (preserving edge sharpness)
- **Video compression** (H.264, HEVC)
- **Signal processing where phase matters** (linear phase property)

**Without tests**, we have no guarantee that:
- Perfect reconstruction is achieved
- Symmetry properties are maintained
- Decomposition and reconstruction filters are correctly paired
- Filter coefficients match published values
- The biorthogonality condition is satisfied

---

## Understanding Biorthogonal Wavelets

### What Makes Biorthogonal Wavelets Special?

Unlike orthogonal wavelets (like Daubechies), biorthogonal wavelets use **different filters for decomposition and reconstruction**. This flexibility allows them to achieve properties impossible with orthogonal wavelets:

1. **Symmetry**: Both scaling and wavelet functions can be symmetric
2. **Linear Phase**: Preserves shape of features (critical for images)
3. **Perfect Reconstruction**: Signal can be perfectly recovered
4. **Flexibility**: Can optimize decomposition and reconstruction separately

### Key Concepts

#### 1. Dual Filter Banks

**Decomposition (Analysis)**:
- Low-pass filter: h (decomposition scaling)
- High-pass filter: g (decomposition wavelet)

**Reconstruction (Synthesis)**:
- Low-pass filter: h~ (reconstruction scaling)
- High-pass filter: g~ (reconstruction wavelet)

**Critical property**: These four filters must satisfy the biorthogonality condition.

#### 2. Biorthogonality Condition

```
<h, h~> = 2
<h, h~(shifted by 2k)> = 0  for k != 0
<h, g~> = 0
<g, h~> = 0
```

**Why it matters**: Ensures perfect reconstruction without redundancy.

#### 3. Linear Phase

**Meaning**: Filter has symmetric impulse response.

**Consequence**: No phase distortion, preserves edges and features.

**Test**: Check if coefficients are symmetric or anti-symmetric.

#### 4. Perfect Reconstruction

**Formula**: `x = Synthesis(Analysis(x))`

**Test**: Decompose then reconstruct, compare with original.

### Common Biorthogonal Wavelets

#### Bior 1.1 (Haar-like)
- Simplest biorthogonal wavelet
- Both filters are the Haar filter
- Symmetric

#### Bior 2.2
- 2 vanishing moments for both decomposition and reconstruction
- Good balance of properties

#### Bior 3.3
- 3 vanishing moments
- Better frequency separation

#### Bior 9/7 (CDF 9/7)
- Used in JPEG2000
- 9-tap analysis filter, 7-tap synthesis filter
- Excellent compression performance

---

## Files Requiring Tests

### 1. BiorthogonalWavelet.cs
**Type**: Biorthogonal, symmetric
**Complexity**: Moderate-High
**Key Properties**:
- Decomposition order (1, 2, 3)
- Reconstruction order (1, 2, 3)
- Separate analysis and synthesis filters
- Linear phase (symmetric)
- Perfect reconstruction

**Current Implementation**: Supports orders 1-3 for both decomposition and reconstruction.

**Test Priority**: CRITICAL (foundation of biorthogonal wavelets)

### 2. ReverseBiorthogonalWavelet.cs
**Type**: Biorthogonal, reversed filter banks
**Complexity**: Moderate
**Key Properties**:
- Swaps the roles of decomposition and reconstruction filters
- Same mathematical properties as BiorthogonalWavelet
- Useful for specific applications

**Test Priority**: HIGH (variant of BiorthogonalWavelet)

### 3. BSplineWavelet.cs
**Type**: Biorthogonal, based on B-spline functions
**Complexity**: Moderate-High
**Key Properties**:
- Order parameter determines B-spline degree
- Smooth scaling functions
- Good approximation properties
- Often symmetric

**Test Priority**: HIGH (important for smooth signal processing)

---

## Testing Strategy

### Test Categories for Biorthogonal Wavelets

#### Category 1: Constructor and Parameter Validation
**Goal**: Ensure wavelets can be created with valid parameter combinations.

```csharp
[Fact]
public void Constructor_DefaultOrders_CreatesValidInstance()
{
    var wavelet = new BiorthogonalWavelet<double>();
    Assert.NotNull(wavelet);
}

[Theory]
[InlineData(1, 1)]  // Bior 1.1 (Haar-like)
[InlineData(1, 3)]  // Bior 1.3
[InlineData(2, 2)]  // Bior 2.2
[InlineData(2, 4)]  // Bior 2.4
[InlineData(3, 3)]  // Bior 3.3
public void Constructor_ValidOrderCombinations_CreatesValidInstance(int decompositionOrder, int reconstructionOrder)
{
    var wavelet = new BiorthogonalWavelet<double>(decompositionOrder, reconstructionOrder);
    Assert.NotNull(wavelet);
}

[Theory]
[InlineData(0, 1)]
[InlineData(1, 0)]
[InlineData(-1, 1)]
[InlineData(4, 2)]  // If only orders 1-3 are supported
public void Constructor_InvalidOrders_ThrowsArgumentException(int decompositionOrder, int reconstructionOrder)
{
    Assert.Throws<ArgumentException>(() =>
        new BiorthogonalWavelet<double>(decompositionOrder, reconstructionOrder));
}
```

#### Category 2: Filter Coefficient Validation
**Goal**: Verify all four filter sets have correct values.

```csharp
[Fact]
public void GetDecompositionLowPassFilter_Bior11_ReturnsCorrectValues()
{
    // Arrange - Bior 1.1 is the Haar wavelet
    var wavelet = new BiorthogonalWavelet<double>(1, 1);
    double expected = Math.Sqrt(2) / 2.0;

    // Act
    var filter = GetDecompositionLowPassFilter(wavelet);

    // Assert
    Assert.Equal(2, filter.Length);
    Assert.Equal(expected, filter[0], precision: 10);
    Assert.Equal(expected, filter[1], precision: 10);
}

[Fact]
public void GetReconstructionLowPassFilter_Bior22_ReturnsCorrectValues()
{
    // Arrange
    var wavelet = new BiorthogonalWavelet<double>(2, 2);

    // Act
    var filter = GetReconstructionLowPassFilter(wavelet);

    // Assert
    Assert.NotNull(filter);
    Assert.True(filter.Length > 0);
    // Verify against published coefficient values for Bior 2.2
}

// Helper methods to access private filters (if needed)
private Vector<double> GetDecompositionLowPassFilter(BiorthogonalWavelet<double> wavelet)
{
    // Use reflection or make filters internal/public for testing
}
```

#### Category 3: Symmetry Tests
**Goal**: Verify filters are symmetric or anti-symmetric (linear phase).

```csharp
[Theory]
[InlineData(1, 1)]
[InlineData(2, 2)]
[InlineData(3, 3)]
public void DecompositionLowPassFilter_IsSymmetric(int decompositionOrder, int reconstructionOrder)
{
    // Arrange
    var wavelet = new BiorthogonalWavelet<double>(decompositionOrder, reconstructionOrder);
    var filter = GetDecompositionLowPassFilter(wavelet);

    // Act & Assert - Check if h[i] == h[n-1-i]
    int n = filter.Length;
    for (int i = 0; i < n / 2; i++)
    {
        Assert.Equal(filter[i], filter[n - 1 - i], precision: 10);
    }
}

[Theory]
[InlineData(1, 1)]
[InlineData(2, 2)]
public void ReconstructionLowPassFilter_IsSymmetric(int decompositionOrder, int reconstructionOrder)
{
    // Arrange
    var wavelet = new BiorthogonalWavelet<double>(decompositionOrder, reconstructionOrder);
    var filter = GetReconstructionLowPassFilter(wavelet);

    // Act & Assert
    int n = filter.Length;
    for (int i = 0; i < n / 2; i++)
    {
        Assert.Equal(filter[i], filter[n - 1 - i], precision: 10);
    }
}
```

#### Category 4: Biorthogonality Condition Tests
**Goal**: Verify the mathematical relationship between analysis and synthesis filters.

```csharp
[Fact]
public void AnalysisAndSynthesisFilters_SatisfyBiorthogonalityCondition()
{
    // Arrange
    var wavelet = new BiorthogonalWavelet<double>(2, 2);
    var h = GetDecompositionLowPassFilter(wavelet);  // Analysis low-pass
    var h_tilde = GetReconstructionLowPassFilter(wavelet);  // Synthesis low-pass

    // Act - Calculate <h, h~> (inner product)
    double innerProduct = 0.0;
    int minLen = Math.Min(h.Length, h_tilde.Length);
    for (int i = 0; i < minLen; i++)
    {
        innerProduct += h[i] * h_tilde[i];
    }

    // Assert - Should equal 2 for biorthogonal wavelets
    Assert.Equal(2.0, innerProduct, precision: 8);
}

[Fact]
public void AnalysisLowPassAndSynthesisHighPass_AreOrthogonal()
{
    // Arrange
    var wavelet = new BiorthogonalWavelet<double>(2, 2);
    var h = GetDecompositionLowPassFilter(wavelet);
    var g_tilde = GetReconstructionHighPassFilter(wavelet);

    // Act - Calculate <h, g~>
    double innerProduct = 0.0;
    int minLen = Math.Min(h.Length, g_tilde.Length);
    for (int i = 0; i < minLen; i++)
    {
        innerProduct += h[i] * g_tilde[i];
    }

    // Assert - Should be zero
    Assert.Equal(0.0, innerProduct, precision: 10);
}

[Fact]
public void ShiftedFilters_AreOrthogonal()
{
    // Arrange
    var wavelet = new BiorthogonalWavelet<double>(2, 2);
    var h = GetDecompositionLowPassFilter(wavelet);
    var h_tilde = GetReconstructionLowPassFilter(wavelet);

    // Act - Check <h, h~(shifted by 2)> = 0
    double innerProduct = 0.0;
    for (int i = 0; i < h.Length - 2; i++)
    {
        innerProduct += h[i] * h_tilde[i + 2];
    }

    // Assert
    Assert.Equal(0.0, innerProduct, precision: 10);
}
```

#### Category 5: Perfect Reconstruction Tests
**Goal**: Verify decomposition followed by reconstruction recovers the original signal.

```csharp
[Theory]
[InlineData(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 })]
[InlineData(new[] { 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0 })]  // Constant
[InlineData(new[] { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 })]  // Impulse
[InlineData(new[] { 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0 })]  // Alternating
public void DecomposeAndReconstruct_VariousSignals_AchievePerfectReconstruction(double[] signalData)
{
    // Arrange
    var wavelet = new BiorthogonalWavelet<double>(2, 2);
    var original = new Vector<double>(signalData);

    // Act
    var (approx, detail) = wavelet.Decompose(original);
    var reconstructed = Reconstruct(wavelet, approx, detail);

    // Assert - Should perfectly recover original
    for (int i = 0; i < original.Length; i++)
    {
        Assert.Equal(original[i], reconstructed[i], precision: 10);
    }
}

[Fact]
public void MultiLevelDecomposition_PerfectReconstruction()
{
    // Arrange
    var wavelet = new BiorthogonalWavelet<double>(2, 2);
    var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

    // Act - Three-level decomposition
    var (approx1, detail1) = wavelet.Decompose(original);
    var (approx2, detail2) = wavelet.Decompose(approx1);
    var (approx3, detail3) = wavelet.Decompose(approx2);

    // Reconstruct in reverse
    var reconstructed2 = Reconstruct(wavelet, approx3, detail3);
    var reconstructed1 = Reconstruct(wavelet, reconstructed2, detail2);
    var final = Reconstruct(wavelet, reconstructed1, detail1);

    // Assert
    for (int i = 0; i < original.Length; i++)
    {
        Assert.Equal(original[i], final[i], precision: 8);
    }
}

private Vector<double> Reconstruct(BiorthogonalWavelet<double> wavelet, Vector<double> approx, Vector<double> detail)
{
    // Implement synthesis using reconstruction filters
    var h_tilde = GetReconstructionLowPassFilter(wavelet);
    var g_tilde = GetReconstructionHighPassFilter(wavelet);

    int outputLength = approx.Length * 2;
    var result = new double[outputLength];

    // Upsample and convolve with synthesis filters
    for (int i = 0; i < approx.Length; i++)
    {
        for (int j = 0; j < h_tilde.Length; j++)
        {
            int idx = (2 * i + j) % outputLength;
            result[idx] += approx[i] * h_tilde[j];
            result[idx] += detail[i] * g_tilde[j];
        }
    }

    return new Vector<double>(result);
}
```

#### Category 6: Decomposition Tests
**Goal**: Verify decomposition produces valid coefficients.

```csharp
[Fact]
public void Decompose_EvenLengthSignal_ProducesHalfLengthOutputs()
{
    // Arrange
    var wavelet = new BiorthogonalWavelet<double>(2, 2);
    var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

    // Act
    var (approx, detail) = wavelet.Decompose(signal);

    // Assert
    Assert.Equal(4, approx.Length);
    Assert.Equal(4, detail.Length);
}

[Fact]
public void Decompose_OddLengthSignal_ThrowsArgumentException()
{
    // Arrange
    var wavelet = new BiorthogonalWavelet<double>(2, 2);
    var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

    // Act & Assert
    Assert.Throws<ArgumentException>(() => wavelet.Decompose(signal));
}

[Fact]
public void Decompose_ProducesFiniteValues()
{
    // Arrange
    var wavelet = new BiorthogonalWavelet<double>(2, 2);
    var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

    // Act
    var (approx, detail) = wavelet.Decompose(signal);

    // Assert - No NaN or Infinity
    Assert.All(approx, value => Assert.False(double.IsNaN(value)));
    Assert.All(approx, value => Assert.False(double.IsInfinity(value)));
    Assert.All(detail, value => Assert.False(double.IsNaN(value)));
    Assert.All(detail, value => Assert.False(double.IsInfinity(value)));
}
```

#### Category 7: Energy Preservation Tests
**Goal**: Verify energy conservation (weaker than orthogonality, but still important).

```csharp
[Theory]
[InlineData(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 })]
[InlineData(new[] { 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0 })]
public void Decompose_PreservesApproximateEnergy(double[] signalData)
{
    // Arrange
    var wavelet = new BiorthogonalWavelet<double>(2, 2);
    var signal = new Vector<double>(signalData);
    double originalEnergy = CalculateEnergy(signal);

    // Act
    var (approx, detail) = wavelet.Decompose(signal);
    double decomposedEnergy = CalculateEnergy(approx) + CalculateEnergy(detail);

    // Assert - Energy should be approximately preserved
    // Note: For biorthogonal wavelets, exact energy preservation is not guaranteed
    // but it should be close
    Assert.Equal(originalEnergy, decomposedEnergy, precision: 6);
}

private double CalculateEnergy(Vector<double> signal)
{
    double energy = 0.0;
    for (int i = 0; i < signal.Length; i++)
    {
        energy += signal[i] * signal[i];
    }
    return energy;
}
```

#### Category 8: Calculate Method Tests
**Goal**: Verify the wavelet function evaluation.

```csharp
[Fact]
public void Calculate_VariousPoints_ReturnsFiniteValues()
{
    // Arrange
    var wavelet = new BiorthogonalWavelet<double>(2, 2);

    // Act & Assert
    for (double x = -5.0; x <= 5.0; x += 0.5)
    {
        double value = wavelet.Calculate(x);
        Assert.False(double.IsNaN(value));
        Assert.False(double.IsInfinity(value));
    }
}

[Fact]
public void Calculate_OutsideSupport_ReturnsZero()
{
    // Arrange
    var wavelet = new BiorthogonalWavelet<double>(2, 2);

    // Act & Assert - Far outside support should be zero
    Assert.Equal(0.0, wavelet.Calculate(-100.0), precision: 10);
    Assert.Equal(0.0, wavelet.Calculate(100.0), precision: 10);
}
```

#### Category 9: Comparison Tests (BiorthogonalWavelet vs ReverseBiorthogonalWavelet)
**Goal**: Verify reversed wavelet swaps filter roles correctly.

```csharp
[Fact]
public void ReverseBiorthogonal_SwapsFilterRoles()
{
    // Arrange
    var bior = new BiorthogonalWavelet<double>(2, 3);
    var rbior = new ReverseBiorthogonalWavelet<double>(2, 3);

    // Act
    var biorDecompLow = GetDecompositionLowPassFilter(bior);
    var rbiorReconLow = GetReconstructionLowPassFilter(rbior);

    // Assert - Reversed should swap decomposition and reconstruction
    // (This depends on actual implementation details)
    Assert.Equal(biorDecompLow.Length, rbiorReconLow.Length);
}
```

#### Category 10: B-Spline Specific Tests
**Goal**: Test B-spline wavelet properties.

```csharp
[Fact]
public void BSplineWavelet_ScalingFunction_IsBSpline()
{
    // Arrange
    var wavelet = new BSplineWavelet<double>(order: 3);

    // Act - Evaluate at several points
    // B-spline scaling functions should be smooth and non-negative

    // Assert - Properties specific to B-splines
}

[Fact]
public void BSplineWavelet_HasCorrectSupport()
{
    // Arrange
    var wavelet = new BSplineWavelet<double>(order: 3);

    // Act & Assert - B-spline of order n has support [0, n+1]
}
```

---

## Step-by-Step Implementation Guide

### Phase 1: Setup Test Infrastructure

#### Step 1: Create Test File Structure
```
tests/
  WaveletFunctions/
    BiorthogonalWavelets/
      BiorthogonalWaveletTests.cs
      ReverseBiorthogonalWaveletTests.cs
      BSplineWaveletTests.cs
```

#### Step 2: Create Shared Helper Class
```csharp
namespace AiDotNet.Tests.WaveletFunctions.BiorthogonalWavelets
{
    public static class BiorthogonalTestHelpers
    {
        public static double CalculateEnergy(Vector<double> signal)
        {
            double energy = 0.0;
            for (int i = 0; i < signal.Length; i++)
            {
                energy += signal[i] * signal[i];
            }
            return energy;
        }

        public static bool IsSymmetric(Vector<double> filter, double tolerance = 1e-10)
        {
            int n = filter.Length;
            for (int i = 0; i < n / 2; i++)
            {
                if (Math.Abs(filter[i] - filter[n - 1 - i]) > tolerance)
                    return false;
            }
            return true;
        }

        public static double CalculateInnerProduct(Vector<double> v1, Vector<double> v2)
        {
            int minLen = Math.Min(v1.Length, v2.Length);
            double product = 0.0;
            for (int i = 0; i < minLen; i++)
            {
                product += v1[i] * v2[i];
            }
            return product;
        }

        public static Vector<double> Reconstruct(
            Vector<double> approx,
            Vector<double> detail,
            Vector<double> h_tilde,
            Vector<double> g_tilde)
        {
            int outputLength = approx.Length * 2;
            var result = new double[outputLength];

            for (int i = 0; i < approx.Length; i++)
            {
                for (int j = 0; j < h_tilde.Length; j++)
                {
                    int idx = (2 * i + j) % outputLength;
                    result[idx] += approx[i] * h_tilde[j];
                }
                for (int j = 0; j < g_tilde.Length; j++)
                {
                    int idx = (2 * i + j) % outputLength;
                    result[idx] += detail[i] * g_tilde[j];
                }
            }

            return new Vector<double>(result);
        }
    }
}
```

### Phase 2: Implement BiorthogonalWavelet Tests

#### Step 1: Start with Simple Constructor Tests
```csharp
[Fact]
public void Constructor_DefaultOrders_CreatesValidInstance()
{
    var wavelet = new BiorthogonalWavelet<double>();
    Assert.NotNull(wavelet);
}
```

#### Step 2: Test All Valid Order Combinations
```csharp
[Theory]
[InlineData(1, 1)]
[InlineData(1, 3)]
[InlineData(2, 2)]
[InlineData(3, 3)]
public void Constructor_ValidOrders_CreatesValidInstance(int d, int r)
{
    var wavelet = new BiorthogonalWavelet<double>(d, r);
    Assert.NotNull(wavelet);
}
```

#### Step 3: Test Coefficient Values
Use published values from PyWavelets or MATLAB.

#### Step 4: Test Perfect Reconstruction
This is the most critical test.

#### Step 5: Test Symmetry
Verify linear phase property.

### Phase 3: Implement ReverseBiorthogonalWavelet Tests

Follow similar structure, but verify filter role swapping.

### Phase 4: Implement BSplineWavelet Tests

Focus on B-spline specific properties.

### Phase 5: Run and Validate

```bash
dotnet test --filter "FullyQualifiedName~BiorthogonalWavelets"
dotnet test /p:CollectCoverage=true
```

---

## Expected Test Structure

### BiorthogonalWaveletTests.cs (Complete Outline)
```csharp
public class BiorthogonalWaveletTests
{
    #region Constructor Tests
    [Fact] public void Constructor_DefaultOrders_CreatesValidInstance() { }
    [Theory] public void Constructor_ValidOrders_CreatesValidInstance(int d, int r) { }
    [Theory] public void Constructor_InvalidOrders_ThrowsException(int d, int r) { }
    #endregion

    #region Coefficient Tests
    [Fact] public void GetDecompositionLowPassFilter_Bior11_CorrectValues() { }
    [Fact] public void GetReconstructionLowPassFilter_Bior22_CorrectValues() { }
    [Fact] public void AllFilters_HaveCorrectLengths() { }
    #endregion

    #region Symmetry Tests
    [Theory] public void DecompositionFilters_AreSymmetric(int d, int r) { }
    [Theory] public void ReconstructionFilters_AreSymmetric(int d, int r) { }
    #endregion

    #region Biorthogonality Tests
    [Fact] public void AnalysisAndSynthesisFilters_SatisfyBiorthogonality() { }
    [Fact] public void CrossFilters_AreOrthogonal() { }
    [Fact] public void ShiftedFilters_AreOrthogonal() { }
    #endregion

    #region Perfect Reconstruction Tests
    [Theory] public void DecomposeReconstruct_VariousSignals_Perfect(double[] data) { }
    [Fact] public void MultiLevelDecomposition_PerfectReconstruction() { }
    #endregion

    #region Decomposition Tests
    [Fact] public void Decompose_EvenLength_ProducesHalfLength() { }
    [Fact] public void Decompose_OddLength_ThrowsException() { }
    [Fact] public void Decompose_ProducesFiniteValues() { }
    #endregion

    #region Energy Tests
    [Theory] public void Decompose_PreservesApproximateEnergy(double[] data) { }
    #endregion

    #region Calculate Tests
    [Fact] public void Calculate_VariousPoints_FiniteValues() { }
    [Fact] public void Calculate_OutsideSupport_ReturnsZero() { }
    #endregion
}
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Confusing Analysis and Synthesis
**Problem**: Mixing up decomposition and reconstruction filters.

**Solution**:
- Clearly label: h (decomp), h~ (recon), g (decomp), g~ (recon)
- Test each filter separately
- Verify perfect reconstruction as integration test

### Pitfall 2: Exact Energy Preservation Assumption
**Problem**: Expecting exact energy preservation like orthogonal wavelets.

**Solution**:
- Biorthogonal wavelets don't preserve energy exactly
- Use looser precision (6 decimal places instead of 10)
- Focus on perfect reconstruction instead

### Pitfall 3: Incorrect Reconstruction Implementation
**Problem**: Synthesis is complex with upsampling and filtering.

**Solution**:
- Study filter bank diagrams
- Test with simple signals (impulse, constant)
- Verify against PyWavelets reference implementation

### Pitfall 4: Boundary Effects
**Problem**: Edge effects in finite signals.

**Solution**:
- Use circular convolution (modulo indexing)
- Test with signals that are powers of 2
- Document boundary mode (periodic, symmetric, etc.)

### Pitfall 5: Not Testing All Order Combinations
**Problem**: Only testing default orders.

**Solution**:
```csharp
[Theory]
[MemberData(nameof(AllValidOrderCombinations))]
public void Test_AllOrders(int d, int r) { }

public static IEnumerable<object[]> AllValidOrderCombinations()
{
    for (int d = 1; d <= 3; d++)
        for (int r = 1; r <= 3; r++)
            yield return new object[] { d, r };
}
```

---

## References

### Mathematical References
1. **Cohen, A., Daubechies, I., & Feauveau, J.-C. (1992)**: "Biorthogonal bases of compactly supported wavelets"
2. **Strang, G. & Nguyen, T.**: "Wavelets and Filter Banks" - Chapter on biorthogonal systems
3. **Vetterli, M. & Kovačević, J.**: "Wavelets and Subband Coding"

### Online Resources
1. **PyWavelets Documentation**: Reference implementation and coefficient tables
2. **JPEG2000 Standard**: Uses CDF 9/7 biorthogonal wavelet
3. **Wavelet Browser**: Interactive visualization of biorthogonal wavelets

### Coefficient References
1. **PyWavelets**: `pywt.wavelist(kind='biorthogonal')` and `pywt.Wavelet('bior2.2').filter_bank`
2. **MATLAB**: `wfilters('bior2.2')` for coefficient values

---

## Success Criteria

Your implementation is complete when:

1. All 3 biorthogonal wavelet classes have comprehensive tests
2. Each file has 20-30 tests covering all categories
3. Code coverage is 80%+ for all files
4. Perfect reconstruction is verified for all order combinations
5. Symmetry properties are confirmed
6. Biorthogonality conditions are satisfied
7. All tests pass consistently
8. Multi-level decomposition works correctly

Good luck!
