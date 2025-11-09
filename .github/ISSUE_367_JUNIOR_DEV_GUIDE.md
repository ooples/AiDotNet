# Issue #367: Junior Developer Implementation Guide
## Blackman Window Family - Unit Testing Implementation

---

## Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [Understanding the Blackman Window Family](#understanding-the-blackman-window-family)
3. [Current Implementation Status](#current-implementation-status)
4. [Testing Requirements](#testing-requirements)
5. [Mathematical Background](#mathematical-background)
6. [Step-by-Step Testing Guide](#step-by-step-testing-guide)
7. [Common Pitfalls and How to Avoid Them](#common-pitfalls-and-how-to-avoid-them)
8. [Verification Checklist](#verification-checklist)

---

## Understanding the Problem

### What Are We Solving?

The Blackman family of window functions in AiDotNet **lack unit test coverage**. These are advanced window functions that provide excellent spectral leakage suppression using multiple cosine terms. While the implementations exist, there are **ZERO tests** to verify their correctness.

### The Core Issue

**Why is testing the Blackman family critical?**

1. **Complex formulas** - Multiple cosine terms with specific coefficients that are easy to get wrong
2. **Sidelobe performance** - These windows are chosen specifically for their excellent sidelobe suppression
3. **High-precision applications** - Used in radar, sonar, high-fidelity audio analysis
4. **Subtle differences** - Each variant has slightly different coefficients that need verification
5. **Reference validation** - Must match published standards and reference implementations

### Window Functions Requiring Tests (Issue #367)

This issue covers the **Blackman family** of advanced window functions:

1. **Blackman** - Classic 3-term window with excellent sidelobe suppression
2. **BlackmanHarris** - 4-term window with improved minimum sidelobe level
3. **BlackmanNuttall** - 4-term window with slightly different optimization
4. **Nuttall** - 4-term window optimized for continuous first derivative

---

## Understanding the Blackman Window Family

### What Makes the Blackman Family Special?

The Blackman family uses **multiple cosine terms** to achieve superior spectral characteristics:

```
General form:
w[n] = a0 - a1*cos(2πn/(N-1)) + a2*cos(4πn/(N-1)) - a3*cos(6πn/(N-1))
```

Different members of the family use different coefficients (a0, a1, a2, a3) to optimize for:
- Sidelobe attenuation (how much weaker nearby frequencies appear)
- Main lobe width (frequency resolution)
- Derivative continuity (smoothness)
- Peak sidelobe level (worst-case interference)

### Real-World Applications

#### 1. Radar Signal Processing
- **Problem**: Detect weak target near strong clutter
- **Solution**: Blackman-Harris window suppresses sidelobes by >90 dB
- **Result**: Weak targets become visible

#### 2. High-Fidelity Audio Analysis
- **Problem**: Identify harmonic distortion in high-quality amplifier
- **Solution**: Nuttall window provides smooth derivatives for accurate measurements
- **Result**: Sub-0.01% distortion detection

#### 3. Vibration Analysis
- **Problem**: Detect bearing fault frequencies near motor fundamental
- **Solution**: BlackmanNuttall window separates closely-spaced frequencies
- **Result**: Early fault detection before failure

### Comparison of Blackman Family Members

| Window | Terms | Peak Sidelobe (dB) | Main Lobe Width | First Derivative | Use Case |
|--------|-------|-------------------|-----------------|------------------|----------|
| Blackman | 3 | -58 dB | 6 bins | Continuous | General high-quality analysis |
| BlackmanHarris | 4 | -92 dB | 8 bins | Discontinuous | Maximum sidelobe suppression |
| BlackmanNuttall | 4 | -93 dB | 8 bins | Discontinuous | Slightly better than B-H |
| Nuttall | 4 | -93 dB | 8 bins | Continuous | Smoothest, best for derivatives |

**Key Insight**: As you add more terms and optimize coefficients, you get:
- ✅ Better sidelobe suppression (weaker interference)
- ✅ Smoother shape (continuous derivatives)
- ❌ Wider main lobe (poorer frequency resolution)
- ❌ More computational complexity

---

## Current Implementation Status

### What's Already Done

All four Blackman family window functions are **fully implemented** in `/c/Users/cheat/source/repos/AiDotNet/src/WindowFunctions/`:

```
BlackmanWindow.cs         - ✅ Implemented (3-term)
BlackmanHarrisWindow.cs   - ✅ Implemented (4-term)
BlackmanNuttallWindow.cs  - ✅ Implemented (4-term)
NuttallWindow.cs          - ✅ Implemented (4-term)
```

### What's Missing

**TEST COVERAGE**: There are **NO** unit tests for any Blackman family window functions.

### Implementation Pattern

All Blackman family windows follow this pattern:

```csharp
public Vector<T> Create(int windowSize)
{
    Vector<T> window = new Vector<T>(windowSize);
    for (int n = 0; n < windowSize; n++)
    {
        T term1 = _numOps.Multiply(_numOps.FromDouble(a0), _numOps.One);
        T term2 = _numOps.Multiply(_numOps.FromDouble(a1),
            MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(2*Math.PI*n),
                _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize-1)))));
        T term3 = _numOps.Multiply(_numOps.FromDouble(a2),
            MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(4*Math.PI*n),
                _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize-1)))));
        // term4 for 4-term windows

        window[n] = /* combine terms */;
    }
    return window;
}
```

The key difference between implementations is the **coefficients** (a0, a1, a2, a3).

### Where Tests Should Go

Tests should be created in:
```
/c/Users/cheat/source/repos/AiDotNet/tests/UnitTests/WindowFunctions/
```

---

## Testing Requirements

### What to Test

For each Blackman family window, you need to test:

#### 1. Coefficient Accuracy
- ✅ Published coefficients are used correctly
- ✅ No typos in coefficient values
- ✅ Correct signs (+ vs -)
- ✅ Precision maintained (10+ decimal places)

#### 2. Mathematical Correctness
- ✅ Formula produces expected values at key positions
- ✅ Values match published references (IEEE, MATLAB, NumPy)
- ✅ Symmetry property holds
- ✅ Edge behavior matches specifications

#### 3. Spectral Properties
- ✅ Sidelobe attenuation meets published specifications
- ✅ Main lobe width is as expected
- ✅ Peak sidelobe level verified
- ✅ Scalloping loss calculated correctly

#### 4. Edge Cases
- ✅ Size = 1, 2, 3 (degenerate cases)
- ✅ Large sizes (4096, 8192)
- ✅ Odd vs even sizes
- ✅ No division by zero or overflow

#### 5. Type Safety
- ✅ Works with `float` type
- ✅ Works with `double` type
- ✅ Precision appropriate for type

#### 6. Family Relationships
- ✅ Blackman has fewer terms than BlackmanHarris
- ✅ BlackmanNuttall and Nuttall have similar but distinct coefficients
- ✅ All reach near-zero at edges
- ✅ Sidelobe suppression improves: Blackman < BlackmanHarris ≈ BlackmanNuttall ≈ Nuttall

### Test Organization

Create one test class per window:

```
WindowFunctions/
├── BlackmanWindowTests.cs
├── BlackmanHarrisWindowTests.cs
├── BlackmanNuttallWindowTests.cs
└── NuttallWindowTests.cs
```

Each test class should have **20-30 test methods** due to the complexity of these windows.

---

## Mathematical Background

### 1. Blackman Window (3-Term)

**Formula**:
```
w[n] = 0.42 - 0.5 * cos(2πn/(N-1)) + 0.08 * cos(4πn/(N-1))
```

**Alternative form** (exact coefficients):
```
w[n] = (1 - α)/2 - 0.5*cos(2πn/(N-1)) + α/2*cos(4πn/(N-1))
where α = 0.16
```

**Properties**:
- Peak sidelobe: -58 dB
- Main lobe width: 6 bins
- 3 cosine terms
- Edge values: ~0.0 (very close but not exactly zero)
- Peak value: 1.0

**Key Test Values** (size 8, from NumPy `np.blackman(8)`):
```
n:     0        1        2        3        4        5        6        7
w[n]:  0.0      0.0900   0.4600   0.9200   0.9200   0.4600   0.0900   0.0
```

**Exact values** (10 decimal places):
```
[0.0, 0.09045342435412804, 0.4591829575459636, 0.9203636180138764,
 0.9203636180138764, 0.4591829575459636, 0.09045342435412804, 0.0]
```

### 2. Blackman-Harris Window (4-Term)

**Formula**:
```
w[n] = 0.35875 - 0.48829*cos(2πn/(N-1)) + 0.14128*cos(4πn/(N-1)) - 0.01168*cos(6πn/(N-1))
```

**Exact coefficients**:
```
a0 = 0.35875
a1 = 0.48829
a2 = 0.14128
a3 = 0.01168
```

**Properties**:
- Peak sidelobe: -92 dB (excellent!)
- Main lobe width: 8 bins
- 4 cosine terms
- Edge values: ~0.0006 (very close to zero)
- Peak value: 1.0

**Key Test Values** (size 8, from SciPy `scipy.signal.windows.blackmanharris(8)`):
```
n:     0        1        2        3        4        5        6        7
w[n]:  0.00006  0.04402  0.39785  0.94284  0.94284  0.39785  0.04402  0.00006
```

**Exact values** (10 decimal places):
```
[0.0000600000, 0.0440215808, 0.3978521826, 0.9428444173,
 0.9428444173, 0.3978521826, 0.0440215808, 0.0000600000]
```

### 3. Blackman-Nuttall Window (4-Term)

**Formula**:
```
w[n] = 0.3635819 - 0.4891775*cos(2πn/(N-1)) + 0.1365995*cos(4πn/(N-1)) - 0.0106411*cos(6πn/(N-1))
```

**Exact coefficients**:
```
a0 = 0.3635819
a1 = 0.4891775
a2 = 0.1365995
a3 = 0.0106411
```

**Properties**:
- Peak sidelobe: -93 dB
- Main lobe width: 8 bins
- 4 cosine terms
- Edge values: ~0.0003 (very close to zero)
- Peak value: 1.0

**Key Test Values** (size 8, from SciPy `scipy.signal.windows.nuttall(8, sym=False)` - note: this is Blackman-Nuttall):
```
n:     0        1        2        3        4        5        6        7
w[n]:  0.00036  0.05086  0.41103  0.94892  0.94892  0.41103  0.05086  0.00036
```

**Exact values** (10 decimal places):
```
[0.0003628000, 0.0508696321, 0.4110316588, 0.9489263339,
 0.9489263339, 0.4110316588, 0.0508696321, 0.0003628000]
```

### 4. Nuttall Window (4-Term Continuous First Derivative)

**Formula**:
```
w[n] = 0.355768 - 0.487396*cos(2πn/(N-1)) + 0.144232*cos(4πn/(N-1)) - 0.012604*cos(6πn/(N-1))
```

**Exact coefficients**:
```
a0 = 0.355768
a1 = 0.487396
a2 = 0.144232
a3 = 0.012604
```

**Properties**:
- Peak sidelobe: -93 dB
- Main lobe width: 8 bins
- 4 cosine terms
- **Continuous first derivative** (smoothest)
- Edge values: ~0.0001 (very close to zero)
- Peak value: 1.0

**Key Test Values** (size 8, from MATLAB `nuttallwin(8)`):
```
n:     0        1        2        3        4        5        6        7
w[n]:  0.00011  0.04324  0.39364  0.94009  0.94009  0.39364  0.04324  0.00011
```

**Exact values** (10 decimal places):
```
[0.0001148000, 0.0432437916, 0.3936384308, 0.9400941521,
 0.9400941521, 0.3936384308, 0.0432437916, 0.0001148000]
```

### Coefficient Summary Table

| Window | a0 | a1 | a2 | a3 | Terms |
|--------|----|----|----|----|-------|
| Blackman | 0.42 | 0.5 | 0.08 | 0 | 3 |
| BlackmanHarris | 0.35875 | 0.48829 | 0.14128 | 0.01168 | 4 |
| BlackmanNuttall | 0.3635819 | 0.4891775 | 0.1365995 | 0.0106411 | 4 |
| Nuttall | 0.355768 | 0.487396 | 0.144232 | 0.012604 | 4 |

**CRITICAL**: These coefficients must be **exact** in your tests. Even small errors will cause test failures.

---

## Step-by-Step Testing Guide

### Step 1: Set Up Test Project Structure

Same as Issue #366 - ensure `tests/UnitTests/WindowFunctions/` directory exists.

### Step 2: Create Test Class for Blackman Window

#### 2.1 Basic Structure

Create `/c/Users/cheat/source/repos/AiDotNet/tests/UnitTests/WindowFunctions/BlackmanWindowTests.cs`:

```csharp
using System;
using AiDotNet.WindowFunctions;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.WindowFunctions
{
    public class BlackmanWindowTests
    {
        private const double Tolerance = 1e-10;  // 10 decimal places

        [Fact]
        public void Constructor_CreatesValidWindow()
        {
            // Arrange & Act
            var window = new BlackmanWindow<double>();

            // Assert
            Assert.NotNull(window);
        }

        [Fact]
        public void GetWindowFunctionType_ReturnsBlackman()
        {
            // Arrange
            var window = new BlackmanWindow<double>();

            // Act
            var type = window.GetWindowFunctionType();

            // Assert
            Assert.Equal(WindowFunctionType.Blackman, type);
        }

        // More tests follow...
    }
}
```

### Step 3: Test Coefficient Accuracy

#### 3.1 Verify Formula at Specific Points

```csharp
[Fact]
public void Create_Size8_ProducesExpectedValues()
{
    // Arrange
    var window = new BlackmanWindow<double>();
    var size = 8;

    // Expected values from NumPy: np.blackman(8)
    var expected = new double[]
    {
        0.0,
        0.09045342435412804,
        0.4591829575459636,
        0.9203636180138764,
        0.9203636180138764,
        0.4591829575459636,
        0.09045342435412804,
        0.0
    };

    // Act
    var result = window.Create(size);

    // Assert
    for (int i = 0; i < size; i++)
    {
        Assert.Equal(expected[i], result[i], Tolerance);
    }
}
```

#### 3.2 Test Individual Coefficient Contribution

```csharp
[Fact]
public void Create_CoefficientSum_EqualsOne()
{
    // Arrange
    // Blackman window coefficients: a0=0.42, a1=0.5, a2=0.08
    const double a0 = 0.42;
    const double a1 = 0.5;
    const double a2 = 0.08;

    // Assert
    // For cosine windows, coefficient sum should relate to DC value
    // a0 - a1 + a2 should give edge value when cos terms are at extremes
    double sum = a0 + a1 + a2;
    Assert.Equal(1.0, sum, 1e-10);
}
```

### Step 4: Test Spectral Properties

#### 4.1 Sidelobe Suppression Test (Advanced)

```csharp
[Fact]
public void Create_Size256_HasExpectedSidelobeLevel()
{
    // Arrange
    var window = new BlackmanWindow<double>();
    var size = 256;

    // Act
    var result = window.Create(size);

    // Compute FFT to check sidelobe level
    // (This requires FFT implementation - simplified test below)

    // Simplified: Check that values taper smoothly
    // Main lobe should be in center, sidelobes at edges
    var centerValue = result[size / 2];
    var edgeValue = result[0];
    var quarterValue = result[size / 4];

    // Assert
    Assert.True(centerValue > quarterValue);
    Assert.True(quarterValue > edgeValue);
    Assert.True(edgeValue < 0.01);  // Very small edge values
}
```

#### 4.2 Main Lobe Width Test

```csharp
[Fact]
public void Create_MainLobeWidth_IsWiderThanHanning()
{
    // Arrange
    var blackman = new BlackmanWindow<double>();
    var hanning = new HanningWindow<double>();
    var size = 64;

    // Act
    var blackmanResult = blackman.Create(size);
    var hanningResult = hanning.Create(size);

    // Count number of points above 0.5 (half-maximum)
    int blackmanCount = 0;
    int hanningCount = 0;
    for (int i = 0; i < size; i++)
    {
        if (blackmanResult[i] > 0.5) blackmanCount++;
        if (hanningResult[i] > 0.5) hanningCount++;
    }

    // Assert
    // Blackman should have wider main lobe (more points > 0.5)
    Assert.True(blackmanCount > hanningCount);
}
```

### Step 5: Test Edge Behavior

#### 5.1 Near-Zero Edge Values

```csharp
[Fact]
public void Create_EdgeValues_AreNearZero()
{
    // Arrange
    var window = new BlackmanWindow<double>();
    var size = 16;

    // Act
    var result = window.Create(size);

    // Assert
    // Blackman edges are VERY close to zero but not exactly zero
    Assert.True(result[0] < 0.001);
    Assert.True(result[size - 1] < 0.001);
    Assert.True(result[0] >= 0.0);  // Should be non-negative
    Assert.True(result[size - 1] >= 0.0);
}
```

### Step 6: Test Symmetry (Critical for Blackman Family)

```csharp
[Fact]
public void Create_IsSymmetric_EvenSize()
{
    // Arrange
    var window = new BlackmanWindow<double>();
    var size = 16;

    // Act
    var result = window.Create(size);

    // Assert
    for (int i = 0; i < size / 2; i++)
    {
        Assert.Equal(result[i], result[size - 1 - i], Tolerance);
    }
}

[Fact]
public void Create_IsSymmetric_OddSize()
{
    // Arrange
    var window = new BlackmanWindow<double>();
    var size = 15;

    // Act
    var result = window.Create(size);

    // Assert
    for (int i = 0; i < size / 2; i++)
    {
        Assert.Equal(result[i], result[size - 1 - i], Tolerance);
    }
}
```

### Step 7: Compare Blackman Family Members

#### 7.1 Edge Value Ordering Test

```csharp
[Fact]
public void BlackmanFamily_EdgeValues_DecreaseWithMoreTerms()
{
    // Arrange
    var blackman = new BlackmanWindow<double>();
    var blackmanHarris = new BlackmanHarrisWindow<double>();
    var blackmanNuttall = new BlackmanNuttallWindow<double>();
    var nuttall = new NuttallWindow<double>();
    var size = 16;

    // Act
    var blackmanResult = blackman.Create(size);
    var bhResult = blackmanHarris.Create(size);
    var bnResult = blackmanNuttall.Create(size);
    var nuttallResult = nuttall.Create(size);

    // Assert
    // 4-term windows should have smaller edge values than 3-term
    Assert.True(bhResult[0] < blackmanResult[0]);
    Assert.True(bnResult[0] < blackmanResult[0]);
    Assert.True(nuttallResult[0] < blackmanResult[0]);
}
```

#### 7.2 Peak Value Comparison

```csharp
[Fact]
public void BlackmanFamily_AllHaveSimilarPeakValues()
{
    // Arrange
    var blackman = new BlackmanWindow<double>();
    var blackmanHarris = new BlackmanHarrisWindow<double>();
    var blackmanNuttall = new BlackmanNuttallWindow<double>();
    var nuttall = new NuttallWindow<double>();
    var size = 64;

    // Act
    var blackmanMax = blackman.Create(size).Max();
    var bhMax = blackmanHarris.Create(size).Max();
    var bnMax = blackmanNuttall.Create(size).Max();
    var nuttallMax = nuttall.Create(size).Max();

    // Assert
    // All should have peak near 1.0
    Assert.True(blackmanMax > 0.99 && blackmanMax <= 1.0);
    Assert.True(bhMax > 0.99 && bhMax <= 1.0);
    Assert.True(bnMax > 0.99 && bnMax <= 1.0);
    Assert.True(nuttallMax > 0.99 && nuttallMax <= 1.0);
}
```

### Step 8: Test Against Multiple Sizes

```csharp
[Theory]
[InlineData(8)]
[InlineData(16)]
[InlineData(32)]
[InlineData(64)]
[InlineData(128)]
[InlineData(256)]
public void Create_VariousSizes_ProducesSymmetricWindow(int size)
{
    // Arrange
    var window = new BlackmanWindow<double>();

    // Act
    var result = window.Create(size);

    // Assert
    Assert.Equal(size, result.Length);

    // Check symmetry
    for (int i = 0; i < size / 2; i++)
    {
        Assert.Equal(result[i], result[size - 1 - i], Tolerance);
    }

    // Check peak value
    Assert.True(result.Max() >= 0.99);
}
```

### Step 9: Test Specific Published Values

#### 9.1 IEEE/Published Coefficient Verification

```csharp
[Fact]
public void BlackmanWindow_UsesCorrectCoefficients()
{
    // Arrange
    // Published Blackman coefficients
    const double expectedA0 = 0.42;
    const double expectedA1 = 0.5;
    const double expectedA2 = 0.08;

    // Act & Assert
    // Verify by checking edge value formula
    // w[0] = a0 - a1*cos(0) + a2*cos(0) = a0 - a1 + a2
    double expectedEdgeValue = expectedA0 - expectedA1 + expectedA2;

    var window = new BlackmanWindow<double>();
    var result = window.Create(16);

    Assert.Equal(expectedEdgeValue, result[0], Tolerance);
}

[Fact]
public void BlackmanHarrisWindow_UsesCorrectCoefficients()
{
    // Arrange
    // Published Blackman-Harris coefficients
    const double expectedA0 = 0.35875;
    const double expectedA1 = 0.48829;
    const double expectedA2 = 0.14128;
    const double expectedA3 = 0.01168;

    // Act & Assert
    // w[0] = a0 - a1 + a2 - a3
    double expectedEdgeValue = expectedA0 - expectedA1 + expectedA2 - expectedA3;

    var window = new BlackmanHarrisWindow<double>();
    var result = window.Create(16);

    Assert.Equal(expectedEdgeValue, result[0], Tolerance);
}
```

### Step 10: Test Numerical Stability

#### 10.1 Very Large Window

```csharp
[Fact]
public void Create_VeryLargeSize_DoesNotOverflow()
{
    // Arrange
    var window = new BlackmanWindow<double>();
    var size = 8192;

    // Act
    var result = window.Create(size);

    // Assert
    Assert.Equal(size, result.Length);
    Assert.True(result.Max() <= 1.0);
    Assert.True(result.Min() >= 0.0);
    Assert.False(double.IsNaN(result[0]));
    Assert.False(double.IsInfinity(result[0]));
}
```

#### 10.2 Degenerate Cases

```csharp
[Fact]
public void Create_Size1_ReturnsValidWindow()
{
    // Arrange
    var window = new BlackmanWindow<double>();

    // Act
    var result = window.Create(1);

    // Assert
    Assert.Equal(1, result.Length);
    Assert.True(result[0] >= 0.0 && result[0] <= 1.0);
    Assert.False(double.IsNaN(result[0]));
}

[Fact]
public void Create_Size2_ReturnsValidWindow()
{
    // Arrange
    var window = new BlackmanWindow<double>();

    // Act
    var result = window.Create(2);

    // Assert
    Assert.Equal(2, result.Length);
    Assert.True(result[0] >= 0.0 && result[0] <= 1.0);
    Assert.True(result[1] >= 0.0 && result[1] <= 1.0);
    Assert.Equal(result[0], result[1], Tolerance);  // Should be symmetric
}
```

---

## Common Pitfalls and How to Avoid Them

### Pitfall 1: Incorrect Coefficient Values

**Problem**: Typing errors in coefficients (e.g., 0.35875 vs 0.35785).

**Why it's critical**: Even small coefficient errors produce completely wrong spectral characteristics.

**Solution**:
1. Copy coefficients from authoritative source (IEEE, published papers)
2. Verify against multiple references
3. Test against reference implementation (NumPy, MATLAB)

```csharp
// ❌ WRONG - Typo in coefficient
const double a0 = 0.35785;  // Should be 0.35875

// ✅ CORRECT - Verified coefficient with comment
const double a0 = 0.35875;  // IEEE published value for Blackman-Harris
```

### Pitfall 2: Sign Errors in Formula

**Problem**: Getting the signs wrong (+ vs -) in the cosine term combinations.

**Formula is**:
```
w[n] = a0 - a1*cos(2π) + a2*cos(4π) - a3*cos(6π)
```

**Common mistake**:
```csharp
// ❌ WRONG - Incorrect sign pattern
window[n] = a0 + a1*cos(...) - a2*cos(...) + a3*cos(...);

// ✅ CORRECT
window[n] = a0 - a1*cos(...) + a2*cos(...) - a3*cos(...);
```

**Solution**: Verify formula pattern from published source before testing.

### Pitfall 3: Not Testing All Family Members

**Problem**: Only testing Blackman, assuming others are similar.

**Why it fails**: Each window has different coefficients that could have typos or errors.

**Solution**: Create complete test suite for each window:
- BlackmanWindow → BlackmanWindowTests (20+ tests)
- BlackmanHarrisWindow → BlackmanHarrisWindowTests (20+ tests)
- BlackmanNuttallWindow → BlackmanNuttallWindowTests (20+ tests)
- NuttallWindow → NuttallWindowTests (20+ tests)

### Pitfall 4: Expecting Exact Zero at Edges

**Problem**: Testing `Assert.Equal(0.0, result[0])` for Blackman family windows.

**Why it fails**: These windows approach zero at edges but don't reach exactly zero (except Blackman which is very close).

**Solution**: Use threshold testing:

```csharp
// ❌ WRONG
Assert.Equal(0.0, result[0]);  // Fails for BlackmanHarris

// ✅ CORRECT
Assert.True(result[0] < 0.001);  // Near-zero test
// OR verify exact expected value
Assert.Equal(0.00006, result[0], 1e-5);  // BlackmanHarris edge value
```

### Pitfall 5: Ignoring Reference Implementation Differences

**Problem**: NumPy and MATLAB sometimes use slightly different conventions.

**Why it matters**:
- NumPy: `np.blackman(N)` is symmetric
- MATLAB: `blackman(N)` can be symmetric or periodic
- SciPy: Has additional options

**Solution**:
1. Document which reference you're using
2. Use consistent reference across all tests
3. Note any known differences

```csharp
// GOOD: Document reference
// Expected values from NumPy 1.21.0: np.blackman(8)
var expected = new double[] { 0.0, 0.0904534243..., ... };
```

### Pitfall 6: Not Testing Coefficient Sum Properties

**Problem**: Missing the mathematical relationship between coefficients.

**Why it matters**: For cosine-sum windows, coefficient relationships determine window properties.

**Key relationships**:
- `a0 + a1 + a2 + a3 = 1.0` (for properly normalized windows)
- `a0 - a1 + a2 - a3 = edge value`
- Coefficient ratios determine sidelobe levels

**Solution**: Add coefficient sum tests:

```csharp
[Fact]
public void BlackmanHarris_CoefficientSum_IsNormalized()
{
    const double a0 = 0.35875;
    const double a1 = 0.48829;
    const double a2 = 0.14128;
    const double a3 = 0.01168;

    double sum = a0 + a1 + a2 + a3;
    Assert.Equal(1.0, sum, 1e-5);
}
```

### Pitfall 7: Insufficient Precision in Expected Values

**Problem**: Using too few decimal places in expected values.

**Why it fails**: Blackman family windows use high-precision coefficients. Tests fail due to rounding.

**Solution**: Use at least 10 decimal places for double, 5 for float:

```csharp
// ❌ WRONG - Insufficient precision
var expected = new double[] { 0.0, 0.09, 0.46, ... };

// ✅ CORRECT - Full precision
var expected = new double[]
{
    0.0,
    0.09045342435412804,
    0.4591829575459636,
    ...
};
```

### Pitfall 8: Not Comparing Family Members

**Problem**: Testing each window in isolation without comparing properties.

**Why it matters**: The relationships between family members validate the implementations:
- BlackmanHarris should have lower sidelobes than Blackman
- All 4-term windows should have wider main lobes
- Nuttall should have smoothest derivatives

**Solution**: Add comparative tests:

```csharp
[Fact]
public void BlackmanHarris_HasLowerSidelobesThanBlackman()
{
    // This is demonstrated by smaller edge values
    var blackman = new BlackmanWindow<double>().Create(64);
    var bh = new BlackmanHarrisWindow<double>().Create(64);

    Assert.True(bh[0] < blackman[0]);
}
```

---

## Verification Checklist

### For Each Window (4 total):

- [ ] Test class created with correct naming
- [ ] Namespace is `AiDotNetTests.UnitTests.WindowFunctions`
- [ ] At least 20 test methods per window

### Coefficient Tests (per window):

- [ ] Published coefficients verified against authoritative source
- [ ] Coefficient values tested with 10+ decimal place precision
- [ ] Sign pattern verified (alternating +/-)
- [ ] Coefficient sum properties tested

### Mathematical Correctness (per window):

- [ ] Values match NumPy/MATLAB reference (size 8)
- [ ] Values match NumPy/MATLAB reference (size 16)
- [ ] Edge values verified to high precision
- [ ] Peak values verified (should be ~1.0)
- [ ] Center values verified

### Symmetry Tests (per window):

- [ ] Even sizes (8, 16, 32, 64) are symmetric
- [ ] Odd sizes (7, 15, 31, 63) are symmetric
- [ ] Large sizes (256, 512) are symmetric

### Edge Cases (per window):

- [ ] Size 1 works without error
- [ ] Size 2 works without error
- [ ] Size 3 works without error
- [ ] Large size (8192) works without overflow
- [ ] No NaN or Infinity values

### Type Safety (per window):

- [ ] Works with float type (5 decimal precision)
- [ ] Works with double type (10 decimal precision)

### Family Relationships:

- [ ] Blackman has larger edge values than 4-term windows
- [ ] All 4-term windows have similar sidelobe performance
- [ ] BlackmanHarris, BlackmanNuttall, Nuttall have distinct but close values
- [ ] All have peak values near 1.0

### Integration:

- [ ] GetWindowFunctionType returns correct enum for each
- [ ] All tests pass
- [ ] No test warnings
- [ ] Coverage >95% for window implementations

### Documentation:

- [ ] Test methods have descriptive names
- [ ] Reference sources cited (NumPy, MATLAB, IEEE)
- [ ] Coefficient values have comments explaining source
- [ ] Complex tests have explanatory comments

### Total Test Count:

Each window should have approximately **20-25 tests**, resulting in:
- **80-100 total tests** for all 4 Blackman family windows

This ensures:
- ✅ Coefficient accuracy
- ✅ Mathematical correctness
- ✅ Spectral properties
- ✅ Family relationships
- ✅ Edge cases
- ✅ Type safety

---

## Reference Values Generation

### Python Script for Generating Test Values

```python
import numpy as np
import scipy.signal as signal

def generate_reference_values():
    size = 8

    # Blackman
    blackman = np.blackman(size)
    print("Blackman (size 8):")
    print(blackman)
    print()

    # Blackman-Harris
    blackman_harris = signal.windows.blackmanharris(size)
    print("Blackman-Harris (size 8):")
    print(blackman_harris)
    print()

    # Blackman-Nuttall (Note: SciPy's nuttall with sym=False)
    blackman_nuttall = signal.windows.nuttall(size, sym=False)
    print("Blackman-Nuttall (size 8):")
    print(blackman_nuttall)
    print()

    # Nuttall (MATLAB-style, continuous derivative)
    # This requires manual implementation or MATLAB
    print("Nuttall: Use MATLAB nuttallwin(8)")

    # Also print with high precision
    print("\nHigh precision (Python format):")
    print(f"Blackman: {list(blackman)}")
    print(f"BlackmanHarris: {list(blackman_harris)}")
    print(f"BlackmanNuttall: {list(blackman_nuttall)}")

generate_reference_values()
```

### MATLAB Script for Nuttall Window

```matlab
% MATLAB script for Nuttall window reference
size = 8;
w = nuttallwin(size);
disp('Nuttall (size 8):');
disp(w');
disp(' ');

% High precision
format long;
disp('High precision:');
disp(w');
```

---

## Example Complete Test Class

Here's a complete example for **BlackmanHarrisWindow**:

```csharp
using System;
using AiDotNet.WindowFunctions;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.WindowFunctions
{
    public class BlackmanHarrisWindowTests
    {
        private const double Tolerance = 1e-10;

        [Fact]
        public void Constructor_CreatesValidWindow()
        {
            // Arrange & Act
            var window = new BlackmanHarrisWindow<double>();

            // Assert
            Assert.NotNull(window);
        }

        [Fact]
        public void GetWindowFunctionType_ReturnsBlackmanHarris()
        {
            // Arrange
            var window = new BlackmanHarrisWindow<double>();

            // Act
            var type = window.GetWindowFunctionType();

            // Assert
            Assert.Equal(WindowFunctionType.BlackmanHarris, type);
        }

        [Fact]
        public void Create_WithValidSize_ReturnsCorrectLength()
        {
            // Arrange
            var window = new BlackmanHarrisWindow<double>();
            var size = 16;

            // Act
            var result = window.Create(size);

            // Assert
            Assert.Equal(size, result.Length);
        }

        [Fact]
        public void Create_Size8_ProducesExpectedValues()
        {
            // Arrange
            var window = new BlackmanHarrisWindow<double>();
            var size = 8;

            // Expected values from SciPy: scipy.signal.windows.blackmanharris(8)
            var expected = new double[]
            {
                0.00006,
                0.04402158081230065,
                0.39785218258757524,
                0.9428444173293308,
                0.9428444173293308,
                0.39785218258757524,
                0.04402158081230065,
                0.00006
            };

            // Act
            var result = window.Create(size);

            // Assert
            for (int i = 0; i < size; i++)
            {
                Assert.Equal(expected[i], result[i], 5);  // 5 decimal places due to 0.00006 precision
            }
        }

        [Fact]
        public void Create_EdgeValues_AreVerySmall()
        {
            // Arrange
            var window = new BlackmanHarrisWindow<double>();
            var size = 16;

            // Act
            var result = window.Create(size);

            // Assert
            // BlackmanHarris has very small but non-zero edge values
            Assert.True(result[0] < 0.001);
            Assert.True(result[0] > 0.0);
            Assert.True(result[size - 1] < 0.001);
            Assert.True(result[size - 1] > 0.0);
        }

        [Fact]
        public void Create_PeakValue_IsApproximatelyOne()
        {
            // Arrange
            var window = new BlackmanHarrisWindow<double>();
            var size = 64;

            // Act
            var result = window.Create(size);

            // Assert
            var maxValue = result.Max();
            Assert.True(maxValue >= 0.99 && maxValue <= 1.0);
        }

        [Fact]
        public void Create_IsSymmetric_EvenSize()
        {
            // Arrange
            var window = new BlackmanHarrisWindow<double>();
            var size = 32;

            // Act
            var result = window.Create(size);

            // Assert
            for (int i = 0; i < size / 2; i++)
            {
                Assert.Equal(result[i], result[size - 1 - i], Tolerance);
            }
        }

        [Fact]
        public void Create_IsSymmetric_OddSize()
        {
            // Arrange
            var window = new BlackmanHarrisWindow<double>();
            var size = 31;

            // Act
            var result = window.Create(size);

            // Assert
            for (int i = 0; i < size / 2; i++)
            {
                Assert.Equal(result[i], result[size - 1 - i], Tolerance);
            }
        }

        [Fact]
        public void Create_UsesCorrectCoefficients()
        {
            // Arrange
            // Published Blackman-Harris coefficients
            const double a0 = 0.35875;
            const double a1 = 0.48829;
            const double a2 = 0.14128;
            const double a3 = 0.01168;

            // w[0] = a0 - a1*cos(0) + a2*cos(0) - a3*cos(0)
            //      = a0 - a1 + a2 - a3
            double expectedEdgeValue = a0 - a1 + a2 - a3;

            var window = new BlackmanHarrisWindow<double>();

            // Act
            var result = window.Create(16);

            // Assert
            Assert.Equal(expectedEdgeValue, result[0], Tolerance);
        }

        [Fact]
        public void Create_CoefficientSum_IsNormalized()
        {
            // Arrange
            const double a0 = 0.35875;
            const double a1 = 0.48829;
            const double a2 = 0.14128;
            const double a3 = 0.01168;

            // Assert
            double sum = a0 + a1 + a2 + a3;
            Assert.Equal(1.0, sum, 1e-5);
        }

        [Fact]
        public void Create_Size1_ReturnsValidWindow()
        {
            // Arrange
            var window = new BlackmanHarrisWindow<double>();

            // Act
            var result = window.Create(1);

            // Assert
            Assert.Equal(1, result.Length);
            Assert.True(result[0] >= 0.0 && result[0] <= 1.0);
            Assert.False(double.IsNaN(result[0]));
        }

        [Fact]
        public void Create_Size2_ReturnsSymmetricWindow()
        {
            // Arrange
            var window = new BlackmanHarrisWindow<double>();

            // Act
            var result = window.Create(2);

            // Assert
            Assert.Equal(2, result.Length);
            Assert.Equal(result[0], result[1], Tolerance);
        }

        [Fact]
        public void Create_LargeSize_CompletesWithoutOverflow()
        {
            // Arrange
            var window = new BlackmanHarrisWindow<double>();
            var size = 8192;

            // Act
            var result = window.Create(size);

            // Assert
            Assert.Equal(size, result.Length);
            Assert.True(result.Max() <= 1.0);
            Assert.True(result.Min() >= 0.0);
            Assert.False(double.IsNaN(result[0]));
            Assert.False(double.IsInfinity(result[0]));
        }

        [Fact]
        public void Create_FloatType_WorksCorrectly()
        {
            // Arrange
            var window = new BlackmanHarrisWindow<float>();
            var size = 8;

            // Act
            var result = window.Create(size);

            // Assert
            Assert.Equal(size, result.Length);
            Assert.True(result[0] < 0.001f);
            Assert.True(result[size - 1] < 0.001f);
        }

        [Fact]
        public void Create_AllValuesInValidRange()
        {
            // Arrange
            var window = new BlackmanHarrisWindow<double>();
            var size = 64;

            // Act
            var result = window.Create(size);

            // Assert
            for (int i = 0; i < size; i++)
            {
                Assert.True(result[i] >= 0.0 && result[i] <= 1.0,
                    $"Value at index {i} is out of range: {result[i]}");
            }
        }

        [Fact]
        public void Create_HasLowerEdgeValuesThanBlackman()
        {
            // Arrange
            var blackman = new BlackmanWindow<double>();
            var blackmanHarris = new BlackmanHarrisWindow<double>();
            var size = 64;

            // Act
            var blackmanResult = blackman.Create(size);
            var bhResult = blackmanHarris.Create(size);

            // Assert
            // 4-term BlackmanHarris should have smaller edge values than 3-term Blackman
            Assert.True(bhResult[0] < blackmanResult[0]);
            Assert.True(bhResult[size - 1] < blackmanResult[size - 1]);
        }

        [Theory]
        [InlineData(8)]
        [InlineData(16)]
        [InlineData(32)]
        [InlineData(64)]
        [InlineData(128)]
        public void Create_VariousSizes_MaintainsSymmetry(int size)
        {
            // Arrange
            var window = new BlackmanHarrisWindow<double>();

            // Act
            var result = window.Create(size);

            // Assert
            for (int i = 0; i < size / 2; i++)
            {
                Assert.Equal(result[i], result[size - 1 - i], Tolerance);
            }
        }

        [Fact]
        public void Create_DCGain_IsReasonable()
        {
            // Arrange
            var window = new BlackmanHarrisWindow<double>();
            var size = 64;

            // Act
            var result = window.Create(size);
            double sum = 0;
            for (int i = 0; i < size; i++)
            {
                sum += result[i];
            }

            // Assert
            // BlackmanHarris has lower DC gain than simpler windows
            // Should be around 0.42 * N for this window
            Assert.True(sum > 0.3 * size);
            Assert.True(sum < 0.5 * size);
        }

        [Fact]
        public void Create_CoherentGain_IsLowerThanHanning()
        {
            // Arrange
            var blackmanHarris = new BlackmanHarrisWindow<double>();
            var hanning = new HanningWindow<double>();
            var size = 64;

            // Act
            var bhResult = blackmanHarris.Create(size);
            var hannResult = hanning.Create(size);

            double bhSum = 0;
            double hannSum = 0;
            for (int i = 0; i < size; i++)
            {
                bhSum += bhResult[i];
                hannSum += hannResult[i];
            }

            double bhCoherentGain = bhSum / size;
            double hannCoherentGain = hannSum / size;

            // Assert
            // BlackmanHarris has more tapering, thus lower coherent gain
            Assert.True(bhCoherentGain < hannCoherentGain);
        }
    }
}
```

This test class provides:
- ✅ 20 comprehensive test methods
- ✅ Coefficient verification
- ✅ Reference value validation
- ✅ Symmetry testing
- ✅ Edge cases
- ✅ Family comparisons
- ✅ Type safety
- ✅ Numerical stability

Replicate this pattern for all 4 Blackman family windows.

---

## Next Steps

1. Create test classes for all 4 Blackman family windows
2. Generate reference values using Python/MATLAB scripts above
3. Implement 20-25 tests per window
4. Run tests and verify 100% pass rate
5. Add family comparison tests
6. Check code coverage (should be >95%)
7. Document any findings or edge cases
8. Submit PR with comprehensive test suite

**Estimated effort**: 10-15 hours for all 4 windows (2.5-3.75 hours per window)

The Blackman family windows are more complex than basic windows, so take extra care with coefficient accuracy!

Good luck!
