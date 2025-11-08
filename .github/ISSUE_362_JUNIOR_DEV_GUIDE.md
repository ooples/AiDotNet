# Issue #362: Junior Developer Implementation Guide
## Implement Tests for Basic Wavelets

---

## Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [Understanding Wavelets](#understanding-wavelets)
3. [Files Requiring Tests](#files-requiring-tests)
4. [Testing Strategy](#testing-strategy)
5. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
6. [Expected Test Structure](#expected-test-structure)
7. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)

---

## Understanding the Problem

### What Are We Solving?

The basic wavelet functions in `src/WaveletFunctions/` currently have **0% test coverage**. We need to achieve **80%+ test coverage** to ensure these critical signal processing components work correctly.

### Why Testing Matters for Wavelets

Wavelets are mathematical functions used for:
- **Signal processing**: Breaking down signals into frequency components
- **Image compression**: JPEG2000, video codecs
- **Feature detection**: Finding edges, patterns, anomalies
- **Data compression**: Removing redundancy while preserving information

**Without tests**, we have no guarantee that:
- The wavelet functions produce mathematically correct values
- Decomposition and reconstruction work properly
- Edge cases are handled correctly
- Changes to the code don't break existing functionality

---

## Understanding Wavelets

### What is a Wavelet?

Think of a wavelet as a **small wave** - a mathematical function that oscillates for a limited time and then dies away. Unlike traditional Fourier transforms that use infinite sine waves, wavelets are **localized in both time and frequency**.

### Key Concepts for Testing

#### 1. Calculate Method
**Purpose**: Evaluates the wavelet function at a specific point.

```csharp
T Calculate(T x)
```

**What to test**:
- Values at known points (compare against mathematical references)
- Behavior at boundaries (edges of the wavelet's support)
- Symmetry properties (if the wavelet should be symmetric)

#### 2. Decompose Method
**Purpose**: Splits a signal into approximation (low-frequency) and detail (high-frequency) components.

```csharp
(Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
```

**What to test**:
- Reconstruction accuracy: Can we rebuild the original signal?
- Energy preservation: Does total energy stay constant?
- Coefficient properties: Are they mathematically valid?

#### 3. GetScalingCoefficients / GetWaveletCoefficients
**Purpose**: Returns the filter coefficients used in decomposition.

```csharp
Vector<T> GetScalingCoefficients()
Vector<T> GetWaveletCoefficients()
```

**What to test**:
- Correct length
- Known coefficient values (from literature)
- Orthogonality (for orthogonal wavelets)
- Normalization (sum of squares)

---

## Files Requiring Tests

### 1. HaarWavelet.cs
**Type**: Orthogonal, compact support
**Complexity**: Simplest wavelet
**Key Properties**:
- Step function: +1 for [0, 0.5), -1 for [0.5, 1), 0 elsewhere
- Scaling coefficients: [1/sqrt(2), 1/sqrt(2)]
- Wavelet coefficients: [1/sqrt(2), -1/sqrt(2)]
- Perfect reconstruction
- Orthogonal

**Test Priority**: HIGH (foundation for understanding other wavelets)

### 2. MexicanHatWavelet.cs
**Type**: Continuous, compact support
**Complexity**: Moderate
**Key Properties**:
- Second derivative of Gaussian
- Formula: (2 - x²/σ²) * exp(-x²/(2σ²))
- Has a central peak and two symmetric valleys
- Not orthogonal (continuous wavelet)
- Admissible (satisfies admissibility condition)

**Test Priority**: HIGH (commonly used in signal analysis)

### 3. ContinuousMexicanHatWavelet.cs
**Type**: Continuous
**Complexity**: Similar to MexicanHatWavelet
**Key Properties**:
- Continuous wavelet transform (CWT) implementation
- Similar mathematical properties to MexicanHatWavelet
- Scale and translation parameters

**Test Priority**: MEDIUM

### 4. MorletWavelet.cs
**Type**: Continuous, Gaussian-modulated
**Complexity**: Moderate
**Key Properties**:
- Gaussian envelope with sinusoidal oscillation
- Formula: exp(iωt) * exp(-t²/(2σ²))
- Excellent time-frequency localization
- Widely used in time-frequency analysis

**Test Priority**: HIGH (very common in practice)

### 5. GaussianWavelet.cs
**Type**: Continuous
**Complexity**: Moderate
**Key Properties**:
- Derivatives of Gaussian function
- Order parameter determines which derivative
- Smooth, well-localized in time and frequency

**Test Priority**: MEDIUM

---

## Testing Strategy

### Test Categories

#### Category 1: Basic Functionality Tests
**Goal**: Ensure the class can be instantiated and basic methods work.

```csharp
[Fact]
public void Constructor_DefaultParameters_DoesNotThrow()
{
    // Arrange & Act
    var wavelet = new HaarWavelet<double>();

    // Assert
    Assert.NotNull(wavelet);
}

[Fact]
public void Constructor_CustomParameters_InitializesCorrectly()
{
    // Arrange & Act
    var wavelet = new MexicanHatWavelet<double>(sigma: 2.0);

    // Assert
    Assert.NotNull(wavelet);
}
```

#### Category 2: Mathematical Correctness Tests
**Goal**: Verify the wavelet function produces mathematically correct values.

```csharp
[Theory]
[InlineData(0.0)]
[InlineData(0.25)]
[InlineData(0.5)]
[InlineData(0.75)]
[InlineData(1.0)]
public void Calculate_KnownPoints_ReturnsExpectedValue(double x)
{
    // Arrange
    var wavelet = new HaarWavelet<double>();
    double expected = CalculateExpectedHaarValue(x);

    // Act
    double actual = wavelet.Calculate(x);

    // Assert
    Assert.Equal(expected, actual, precision: 10);
}

private double CalculateExpectedHaarValue(double x)
{
    if (x >= 0.0 && x < 0.5) return 1.0;
    if (x >= 0.5 && x < 1.0) return -1.0;
    return 0.0;
}
```

#### Category 3: Decomposition Tests
**Goal**: Verify signal decomposition works correctly.

```csharp
[Fact]
public void Decompose_SimpleSignal_ProducesValidCoefficients()
{
    // Arrange
    var wavelet = new HaarWavelet<double>();
    var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

    // Act
    var (approx, detail) = wavelet.Decompose(signal);

    // Assert
    Assert.Equal(2, approx.Length);  // Half length due to downsampling
    Assert.Equal(2, detail.Length);
    Assert.All(approx, value => Assert.False(double.IsNaN(value)));
    Assert.All(detail, value => Assert.False(double.IsNaN(value)));
}

[Fact]
public void Decompose_OddLength_ThrowsArgumentException()
{
    // Arrange
    var wavelet = new HaarWavelet<double>();
    var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0 });  // Odd length

    // Act & Assert
    Assert.Throws<ArgumentException>(() => wavelet.Decompose(signal));
}
```

#### Category 4: Reconstruction Tests
**Goal**: Verify perfect reconstruction (for orthogonal wavelets).

```csharp
[Fact]
public void Decompose_Reconstruct_RecoverOriginalSignal()
{
    // Arrange
    var wavelet = new HaarWavelet<double>();
    var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

    // Act
    var (approx, detail) = wavelet.Decompose(original);
    var reconstructed = Reconstruct(approx, detail, wavelet);

    // Assert
    for (int i = 0; i < original.Length; i++)
    {
        Assert.Equal(original[i], reconstructed[i], precision: 10);
    }
}

private Vector<double> Reconstruct(Vector<double> approx, Vector<double> detail, HaarWavelet<double> wavelet)
{
    // Implement inverse wavelet transform
    // This tests that decomposition is reversible
}
```

#### Category 5: Coefficient Validation Tests
**Goal**: Verify filter coefficients match known values.

```csharp
[Fact]
public void GetScalingCoefficients_HaarWavelet_ReturnsCorrectValues()
{
    // Arrange
    var wavelet = new HaarWavelet<double>();
    double expectedValue = 1.0 / Math.Sqrt(2.0);

    // Act
    var coeffs = wavelet.GetScalingCoefficients();

    // Assert
    Assert.Equal(2, coeffs.Length);
    Assert.Equal(expectedValue, coeffs[0], precision: 10);
    Assert.Equal(expectedValue, coeffs[1], precision: 10);
}

[Fact]
public void GetWaveletCoefficients_HaarWavelet_ReturnsCorrectValues()
{
    // Arrange
    var wavelet = new HaarWavelet<double>();
    double expectedPositive = 1.0 / Math.Sqrt(2.0);
    double expectedNegative = -1.0 / Math.Sqrt(2.0);

    // Act
    var coeffs = wavelet.GetWaveletCoefficients();

    // Assert
    Assert.Equal(2, coeffs.Length);
    Assert.Equal(expectedPositive, coeffs[0], precision: 10);
    Assert.Equal(expectedNegative, coeffs[1], precision: 10);
}
```

#### Category 6: Orthogonality Tests (for orthogonal wavelets only)
**Goal**: Verify orthogonality properties.

```csharp
[Fact]
public void Coefficients_Orthogonality_SatisfiesOrthogonalityCondition()
{
    // Arrange
    var wavelet = new HaarWavelet<double>();
    var h = wavelet.GetScalingCoefficients();
    var g = wavelet.GetWaveletCoefficients();

    // Act
    double innerProduct = 0.0;
    for (int i = 0; i < h.Length; i++)
    {
        innerProduct += h[i] * g[i];
    }

    // Assert
    Assert.Equal(0.0, innerProduct, precision: 10);  // Should be zero for orthogonal wavelets
}
```

#### Category 7: Energy Preservation Tests
**Goal**: Verify Parseval's theorem (energy conservation).

```csharp
[Fact]
public void Decompose_EnergyPreservation_MaintainsSignalEnergy()
{
    // Arrange
    var wavelet = new HaarWavelet<double>();
    var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
    double originalEnergy = CalculateEnergy(signal);

    // Act
    var (approx, detail) = wavelet.Decompose(signal);
    double decomposedEnergy = CalculateEnergy(approx) + CalculateEnergy(detail);

    // Assert
    Assert.Equal(originalEnergy, decomposedEnergy, precision: 10);
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

#### Category 8: Edge Case Tests
**Goal**: Handle boundary conditions and invalid inputs.

```csharp
[Fact]
public void Calculate_OutsideSupport_ReturnsZero()
{
    // Arrange
    var wavelet = new HaarWavelet<double>();

    // Act & Assert
    Assert.Equal(0.0, wavelet.Calculate(-1.0));
    Assert.Equal(0.0, wavelet.Calculate(2.0));
}

[Fact]
public void Decompose_EmptyVector_ThrowsArgumentException()
{
    // Arrange
    var wavelet = new HaarWavelet<double>();
    var empty = new Vector<double>(0);

    // Act & Assert
    Assert.Throws<ArgumentException>(() => wavelet.Decompose(empty));
}
```

---

## Step-by-Step Implementation Guide

### Phase 1: Setup Test Infrastructure

#### Step 1: Create Test File Structure
```
tests/
  WaveletFunctions/
    BasicWavelets/
      HaarWaveletTests.cs
      MexicanHatWaveletTests.cs
      ContinuousMexicanHatWaveletTests.cs
      MorletWaveletTests.cs
      GaussianWaveletTests.cs
```

#### Step 2: Set Up Base Test Class (Optional)
```csharp
using Xunit;
using AiDotNet.WaveletFunctions;
using AiDotNet.DataStructures;

namespace AiDotNet.Tests.WaveletFunctions.BasicWavelets
{
    public abstract class WaveletTestBase<T>
    {
        protected abstract IWaveletFunction<T> CreateWavelet();

        protected double CalculateEnergy(Vector<T> signal)
        {
            // Common energy calculation logic
        }

        protected bool AreVectorsEqual(Vector<T> v1, Vector<T> v2, double tolerance = 1e-10)
        {
            // Common comparison logic
        }
    }
}
```

### Phase 2: Implement Tests for HaarWavelet (Start Here)

**Why Haar First?**
- Simplest wavelet
- Easy to verify manually
- Foundation for understanding other wavelets
- Has exact mathematical values (no approximations)

#### Step 1: Create HaarWaveletTests.cs
```csharp
using Xunit;
using AiDotNet.WaveletFunctions;
using AiDotNet.DataStructures;
using System;

namespace AiDotNet.Tests.WaveletFunctions.BasicWavelets
{
    public class HaarWaveletTests
    {
        // Tests go here
    }
}
```

#### Step 2: Write Constructor Tests
```csharp
[Fact]
public void Constructor_DefaultParameters_CreatesValidInstance()
{
    // Arrange & Act
    var wavelet = new HaarWavelet<double>();

    // Assert
    Assert.NotNull(wavelet);
}
```

#### Step 3: Write Calculate Tests
```csharp
[Theory]
[InlineData(0.0, 1.0)]      // Left boundary, should be +1
[InlineData(0.25, 1.0)]     // First quarter, should be +1
[InlineData(0.499, 1.0)]    // Just before midpoint, should be +1
[InlineData(0.5, -1.0)]     // Midpoint, should be -1
[InlineData(0.75, -1.0)]    // Third quarter, should be -1
[InlineData(0.999, -1.0)]   // Just before right boundary, should be -1
[InlineData(1.0, 0.0)]      // Right boundary, should be 0
[InlineData(-0.5, 0.0)]     // Before support, should be 0
[InlineData(1.5, 0.0)]      // After support, should be 0
public void Calculate_VariousPoints_ReturnsExpectedValue(double x, double expected)
{
    // Arrange
    var wavelet = new HaarWavelet<double>();

    // Act
    double actual = wavelet.Calculate(x);

    // Assert
    Assert.Equal(expected, actual, precision: 10);
}
```

#### Step 4: Write Coefficient Tests
```csharp
[Fact]
public void GetScalingCoefficients_ReturnsCorrectLength()
{
    // Arrange
    var wavelet = new HaarWavelet<double>();

    // Act
    var coeffs = wavelet.GetScalingCoefficients();

    // Assert
    Assert.Equal(2, coeffs.Length);
}

[Fact]
public void GetScalingCoefficients_ReturnsNormalizedValues()
{
    // Arrange
    var wavelet = new HaarWavelet<double>();
    double expected = 1.0 / Math.Sqrt(2.0);  // ≈ 0.7071

    // Act
    var coeffs = wavelet.GetScalingCoefficients();

    // Assert
    Assert.Equal(expected, coeffs[0], precision: 10);
    Assert.Equal(expected, coeffs[1], precision: 10);
}

[Fact]
public void GetWaveletCoefficients_ReturnsCorrectLength()
{
    // Arrange
    var wavelet = new HaarWavelet<double>();

    // Act
    var coeffs = wavelet.GetWaveletCoefficients();

    // Assert
    Assert.Equal(2, coeffs.Length);
}

[Fact]
public void GetWaveletCoefficients_ReturnsNormalizedValues()
{
    // Arrange
    var wavelet = new HaarWavelet<double>();
    double expectedPositive = 1.0 / Math.Sqrt(2.0);
    double expectedNegative = -1.0 / Math.Sqrt(2.0);

    // Act
    var coeffs = wavelet.GetWaveletCoefficients();

    // Assert
    Assert.Equal(expectedPositive, coeffs[0], precision: 10);
    Assert.Equal(expectedNegative, coeffs[1], precision: 10);
}
```

#### Step 5: Write Decomposition Tests
```csharp
[Fact]
public void Decompose_EvenLengthSignal_ProducesHalfLengthOutputs()
{
    // Arrange
    var wavelet = new HaarWavelet<double>();
    var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

    // Act
    var (approx, detail) = wavelet.Decompose(signal);

    // Assert
    Assert.Equal(2, approx.Length);
    Assert.Equal(2, detail.Length);
}

[Fact]
public void Decompose_OddLengthSignal_ThrowsArgumentException()
{
    // Arrange
    var wavelet = new HaarWavelet<double>();
    var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

    // Act & Assert
    var exception = Assert.Throws<ArgumentException>(() => wavelet.Decompose(signal));
    Assert.Contains("even", exception.Message.ToLower());
}

[Fact]
public void Decompose_KnownSignal_ProducesExpectedCoefficients()
{
    // Arrange
    var wavelet = new HaarWavelet<double>();
    var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

    // Expected values (manually calculated):
    // Approx: [(1+2)/sqrt(2), (3+4)/sqrt(2)] = [2.121, 4.950]
    // Detail: [(1-2)/sqrt(2), (3-4)/sqrt(2)] = [-0.707, -0.707]
    double sqrt2 = Math.Sqrt(2.0);

    // Act
    var (approx, detail) = wavelet.Decompose(signal);

    // Assert
    Assert.Equal(3.0 / sqrt2, approx[0], precision: 10);
    Assert.Equal(7.0 / sqrt2, approx[1], precision: 10);
    Assert.Equal(-1.0 / sqrt2, detail[0], precision: 10);
    Assert.Equal(-1.0 / sqrt2, detail[1], precision: 10);
}
```

#### Step 6: Write Energy Preservation Tests
```csharp
[Theory]
[InlineData(new[] { 1.0, 2.0, 3.0, 4.0 })]
[InlineData(new[] { 5.0, 5.0, 5.0, 5.0 })]
[InlineData(new[] { 1.0, -1.0, 1.0, -1.0 })]
[InlineData(new[] { 0.0, 0.0, 0.0, 0.0 })]
public void Decompose_PreservesSignalEnergy(double[] signalData)
{
    // Arrange
    var wavelet = new HaarWavelet<double>();
    var signal = new Vector<double>(signalData);
    double originalEnergy = CalculateEnergy(signal);

    // Act
    var (approx, detail) = wavelet.Decompose(signal);
    double decomposedEnergy = CalculateEnergy(approx) + CalculateEnergy(detail);

    // Assert
    Assert.Equal(originalEnergy, decomposedEnergy, precision: 10);
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

#### Step 7: Write Orthogonality Tests
```csharp
[Fact]
public void ScalingAndWaveletCoefficients_AreOrthogonal()
{
    // Arrange
    var wavelet = new HaarWavelet<double>();
    var h = wavelet.GetScalingCoefficients();
    var g = wavelet.GetWaveletCoefficients();

    // Act
    double innerProduct = 0.0;
    for (int i = 0; i < h.Length; i++)
    {
        innerProduct += h[i] * g[i];
    }

    // Assert
    Assert.Equal(0.0, innerProduct, precision: 10);
}

[Fact]
public void ScalingCoefficients_AreNormalized()
{
    // Arrange
    var wavelet = new HaarWavelet<double>();
    var h = wavelet.GetScalingCoefficients();

    // Act
    double sumOfSquares = 0.0;
    for (int i = 0; i < h.Length; i++)
    {
        sumOfSquares += h[i] * h[i];
    }

    // Assert
    Assert.Equal(1.0, sumOfSquares, precision: 10);
}

[Fact]
public void WaveletCoefficients_AreNormalized()
{
    // Arrange
    var wavelet = new HaarWavelet<double>();
    var g = wavelet.GetWaveletCoefficients();

    // Act
    double sumOfSquares = 0.0;
    for (int i = 0; i < g.Length; i++)
    {
        sumOfSquares += g[i] * g[i];
    }

    // Assert
    Assert.Equal(1.0, sumOfSquares, precision: 10);
}
```

### Phase 3: Implement Tests for MexicanHatWavelet

#### Step 1: Create MexicanHatWaveletTests.cs

#### Step 2: Write Constructor Tests (with parameter)
```csharp
[Fact]
public void Constructor_DefaultSigma_CreatesValidInstance()
{
    // Arrange & Act
    var wavelet = new MexicanHatWavelet<double>();

    // Assert
    Assert.NotNull(wavelet);
}

[Theory]
[InlineData(0.5)]
[InlineData(1.0)]
[InlineData(2.0)]
[InlineData(5.0)]
public void Constructor_CustomSigma_CreatesValidInstance(double sigma)
{
    // Arrange & Act
    var wavelet = new MexicanHatWavelet<double>(sigma);

    // Assert
    Assert.NotNull(wavelet);
}
```

#### Step 3: Write Calculate Tests (Formula-based)
```csharp
[Theory]
[InlineData(0.0, 1.0, 2.0)]       // At center (x=0), sigma=1, expect 2
[InlineData(1.0, 1.0)]            // At x=1, sigma=1
[InlineData(-1.0, 1.0)]           // Symmetry test
public void Calculate_KnownPoints_ReturnsExpectedValue(double x, double sigma, double? expectedValue = null)
{
    // Arrange
    var wavelet = new MexicanHatWavelet<double>(sigma);
    double expected = expectedValue ?? CalculateExpectedMexicanHat(x, sigma);

    // Act
    double actual = wavelet.Calculate(x);

    // Assert
    Assert.Equal(expected, actual, precision: 6);  // Less precision due to floating-point
}

private double CalculateExpectedMexicanHat(double x, double sigma)
{
    // Formula: (2 - x²/σ²) * exp(-x²/(2σ²))
    double x2 = x * x;
    double sigma2 = sigma * sigma;
    double term1 = 2.0 - (x2 / sigma2);
    double term2 = Math.Exp(-x2 / (2.0 * sigma2));
    return term1 * term2;
}
```

#### Step 4: Write Symmetry Tests
```csharp
[Theory]
[InlineData(1.0, 1.0)]
[InlineData(2.0, 1.0)]
[InlineData(0.5, 2.0)]
public void Calculate_SymmetricPoints_ProducesEqualValues(double x, double sigma)
{
    // Arrange
    var wavelet = new MexicanHatWavelet<double>(sigma);

    // Act
    double positiveValue = wavelet.Calculate(x);
    double negativeValue = wavelet.Calculate(-x);

    // Assert
    Assert.Equal(positiveValue, negativeValue, precision: 10);
}
```

#### Step 5: Write Admissibility Tests
```csharp
[Fact]
public void Calculate_IntegralOfWavelet_IsZero()
{
    // Arrange
    var wavelet = new MexicanHatWavelet<double>(1.0);
    double integral = 0.0;
    double step = 0.01;

    // Act - Numerical integration
    for (double x = -10.0; x <= 10.0; x += step)
    {
        integral += wavelet.Calculate(x) * step;
    }

    // Assert - Should be close to zero (admissibility condition)
    Assert.Equal(0.0, integral, precision: 2);
}
```

#### Step 6: Write Decomposition Tests
```csharp
[Fact]
public void Decompose_SimpleSignal_ProducesValidCoefficients()
{
    // Arrange
    var wavelet = new MexicanHatWavelet<double>();
    var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

    // Act
    var (approx, detail) = wavelet.Decompose(signal);

    // Assert
    Assert.Equal(signal.Length, approx.Length);  // Same length for continuous wavelets
    Assert.Equal(signal.Length, detail.Length);
    Assert.All(approx, value => Assert.False(double.IsNaN(value)));
    Assert.All(detail, value => Assert.False(double.IsNaN(value)));
}
```

### Phase 4: Implement Tests for Remaining Wavelets

Follow the same pattern for:
- **ContinuousMexicanHatWavelet**: Similar to MexicanHat
- **MorletWavelet**: Test with omega and sigma parameters
- **GaussianWavelet**: Test different derivative orders

### Phase 5: Run and Validate Tests

#### Step 1: Run All Tests
```bash
dotnet test --filter "FullyQualifiedName~WaveletFunctions.BasicWavelets"
```

#### Step 2: Check Coverage
```bash
dotnet test /p:CollectCoverage=true /p:CoverletOutputFormat=opencover
```

#### Step 3: Aim for 80%+ Coverage
Ensure you have:
- Constructor tests
- Calculate tests with various inputs
- Coefficient validation tests
- Decomposition tests
- Property tests (orthogonality, energy, etc.)
- Edge case tests

---

## Expected Test Structure

### File: HaarWaveletTests.cs (Complete Example)
```csharp
using Xunit;
using AiDotNet.WaveletFunctions;
using AiDotNet.DataStructures;
using System;

namespace AiDotNet.Tests.WaveletFunctions.BasicWavelets
{
    public class HaarWaveletTests
    {
        #region Constructor Tests

        [Fact]
        public void Constructor_DefaultParameters_CreatesValidInstance()
        {
            var wavelet = new HaarWavelet<double>();
            Assert.NotNull(wavelet);
        }

        #endregion

        #region Calculate Tests

        [Theory]
        [InlineData(0.0, 1.0)]
        [InlineData(0.25, 1.0)]
        [InlineData(0.5, -1.0)]
        [InlineData(0.75, -1.0)]
        [InlineData(1.0, 0.0)]
        [InlineData(-0.5, 0.0)]
        [InlineData(1.5, 0.0)]
        public void Calculate_VariousPoints_ReturnsExpectedValue(double x, double expected)
        {
            var wavelet = new HaarWavelet<double>();
            double actual = wavelet.Calculate(x);
            Assert.Equal(expected, actual, precision: 10);
        }

        #endregion

        #region Coefficient Tests

        [Fact]
        public void GetScalingCoefficients_ReturnsCorrectValues()
        {
            var wavelet = new HaarWavelet<double>();
            var coeffs = wavelet.GetScalingCoefficients();
            double expected = 1.0 / Math.Sqrt(2.0);

            Assert.Equal(2, coeffs.Length);
            Assert.Equal(expected, coeffs[0], precision: 10);
            Assert.Equal(expected, coeffs[1], precision: 10);
        }

        [Fact]
        public void GetWaveletCoefficients_ReturnsCorrectValues()
        {
            var wavelet = new HaarWavelet<double>();
            var coeffs = wavelet.GetWaveletCoefficients();
            double expectedPos = 1.0 / Math.Sqrt(2.0);
            double expectedNeg = -1.0 / Math.Sqrt(2.0);

            Assert.Equal(2, coeffs.Length);
            Assert.Equal(expectedPos, coeffs[0], precision: 10);
            Assert.Equal(expectedNeg, coeffs[1], precision: 10);
        }

        #endregion

        #region Decomposition Tests

        [Fact]
        public void Decompose_EvenLengthSignal_ProducesHalfLengthOutputs()
        {
            var wavelet = new HaarWavelet<double>();
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

            var (approx, detail) = wavelet.Decompose(signal);

            Assert.Equal(2, approx.Length);
            Assert.Equal(2, detail.Length);
        }

        [Fact]
        public void Decompose_OddLengthSignal_ThrowsArgumentException()
        {
            var wavelet = new HaarWavelet<double>();
            var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            Assert.Throws<ArgumentException>(() => wavelet.Decompose(signal));
        }

        #endregion

        #region Energy Preservation Tests

        [Theory]
        [InlineData(new[] { 1.0, 2.0, 3.0, 4.0 })]
        [InlineData(new[] { 5.0, 5.0, 5.0, 5.0 })]
        public void Decompose_PreservesSignalEnergy(double[] signalData)
        {
            var wavelet = new HaarWavelet<double>();
            var signal = new Vector<double>(signalData);
            double originalEnergy = CalculateEnergy(signal);

            var (approx, detail) = wavelet.Decompose(signal);
            double decomposedEnergy = CalculateEnergy(approx) + CalculateEnergy(detail);

            Assert.Equal(originalEnergy, decomposedEnergy, precision: 10);
        }

        #endregion

        #region Orthogonality Tests

        [Fact]
        public void ScalingAndWaveletCoefficients_AreOrthogonal()
        {
            var wavelet = new HaarWavelet<double>();
            var h = wavelet.GetScalingCoefficients();
            var g = wavelet.GetWaveletCoefficients();

            double innerProduct = 0.0;
            for (int i = 0; i < h.Length; i++)
            {
                innerProduct += h[i] * g[i];
            }

            Assert.Equal(0.0, innerProduct, precision: 10);
        }

        #endregion

        #region Helper Methods

        private double CalculateEnergy(Vector<double> signal)
        {
            double energy = 0.0;
            for (int i = 0; i < signal.Length; i++)
            {
                energy += signal[i] * signal[i];
            }
            return energy;
        }

        #endregion
    }
}
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Floating-Point Precision Issues
**Problem**: Tests fail due to tiny rounding errors.

**Solution**: Use appropriate precision in assertions.
```csharp
// BAD
Assert.Equal(expected, actual);

// GOOD
Assert.Equal(expected, actual, precision: 10);  // 10 decimal places
```

### Pitfall 2: Not Testing Edge Cases
**Problem**: Missing tests for boundary conditions.

**Solution**: Always test:
- Empty inputs
- Single-element inputs
- Boundary values (0, 1, -1, etc.)
- Values outside the wavelet's support

### Pitfall 3: Incorrect Expected Values
**Problem**: Manual calculations are wrong.

**Solution**:
- Use reference implementations (SciPy, PyWavelets)
- Double-check formulas against literature
- Use symbolic math tools (Mathematica, SymPy)

### Pitfall 4: Forgetting Continuous vs Discrete Wavelets
**Problem**: Applying discrete wavelet tests to continuous wavelets.

**Solution**:
- Discrete wavelets (Haar): Downsampling, perfect reconstruction
- Continuous wavelets (Mexican Hat, Morlet): No downsampling, approximate reconstruction

### Pitfall 5: Not Testing All Generic Types
**Problem**: Only testing with `double`, ignoring `float`.

**Solution**: Test both if supported:
```csharp
[Fact]
public void Constructor_FloatType_CreatesValidInstance()
{
    var wavelet = new HaarWavelet<float>();
    Assert.NotNull(wavelet);
}
```

### Pitfall 6: Ignoring Admissibility Condition
**Problem**: Not testing that wavelets satisfy the admissibility condition.

**Solution**: Test that integral of wavelet is zero:
```csharp
[Fact]
public void Calculate_IntegralOfWavelet_IsZero()
{
    // Numerical integration test
}
```

---

## References

### Mathematical References
1. **"Ten Lectures on Wavelets"** by Ingrid Daubechies - The definitive wavelet textbook
2. **"A Wavelet Tour of Signal Processing"** by Stéphane Mallat - Comprehensive signal processing perspective
3. **"Wavelets and Filter Banks"** by Strang & Nguyen - Practical implementation guide

### Online Resources
1. **PyWavelets Documentation**: Reference implementation for expected values
2. **Wikipedia - Wavelet**: Good overview of different wavelet families
3. **Wolfram MathWorld - Wavelet**: Mathematical definitions and formulas

### Testing Best Practices
1. **xUnit Documentation**: https://xunit.net/
2. **Arrange-Act-Assert Pattern**: Standard unit test structure
3. **Theory vs Fact**: When to use parameterized tests

---

## Success Criteria

Your implementation is complete when:

1. All 5 wavelet classes have test files
2. Each test file has at least 15-20 tests
3. Code coverage is 80%+ for all files
4. All tests pass consistently
5. Tests cover:
   - Constructor validation
   - Calculate method correctness
   - Coefficient validation
   - Decomposition functionality
   - Mathematical properties (orthogonality, energy preservation, etc.)
   - Edge cases and error handling

---

## Getting Help

If you get stuck:
1. Review the existing test files in the codebase
2. Check the wavelet implementation for expected behavior
3. Consult mathematical references for correct formulas
4. Ask for help with specific failing tests
5. Use reference implementations (PyWavelets) to verify expected values

Good luck with the implementation!
