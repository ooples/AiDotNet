# Issue #366: Junior Developer Implementation Guide
## Basic Window Functions - Unit Testing Implementation

---

## Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [Understanding Window Functions](#understanding-window-functions)
3. [Current Implementation Status](#current-implementation-status)
4. [Testing Requirements](#testing-requirements)
5. [Mathematical Background](#mathematical-background)
6. [Step-by-Step Testing Guide](#step-by-step-testing-guide)
7. [Common Pitfalls and How to Avoid Them](#common-pitfalls-and-how-to-avoid-them)
8. [Verification Checklist](#verification-checklist)

---

## Understanding the Problem

### What Are We Solving?

The basic window functions in AiDotNet **lack unit test coverage**. While the implementations exist and appear correct, there are **ZERO tests** to verify their mathematical correctness, edge case handling, and integration with the rest of the system.

### The Core Issue

**Why is 0% test coverage a problem?**

1. **No verification** that the mathematical formulas are implemented correctly
2. **No confidence** that edge cases (size 1, size 2, very large sizes) are handled
3. **No regression testing** to catch bugs introduced by future changes
4. **No documentation** of expected behavior through test examples
5. **No validation** of symmetry, normalization, and other window properties

### Window Functions Requiring Tests (Issue #366)

This issue covers the **basic** window functions:

1. **Rectangular** - Simplest window (all values = 1)
2. **Triangular** - Linear ramp up and down
3. **Hanning** - Raised cosine (one term)
4. **Hamming** - Raised cosine with optimized coefficients
5. **Bartlett** - Triangular variant reaching zero at edges
6. **BartlettHann** - Hybrid of Bartlett and Hann
7. **Cosine** - Simple sine-based window
8. **Welch** - Parabolic window

---

## Understanding Window Functions

### What is a Window Function?

A **window function** is a mathematical function that is zero-valued outside of a chosen interval. In signal processing, window functions are used to:

1. **Reduce spectral leakage** when analyzing finite segments of longer signals
2. **Smooth transitions** at segment boundaries to minimize artifacts
3. **Control trade-offs** between frequency resolution and spectral leakage

### Real-World Analogy

Imagine you're analyzing a continuous music stream but can only listen to 10-second chunks:

- **Rectangular window**: Suddenly start and stop listening (creates "click" artifacts)
- **Hanning window**: Gradually increase and decrease listening volume (smoother)
- **Hamming window**: Similar to Hanning but with optimized fade-in/fade-out

### Key Window Properties

All window functions in this issue share these characteristics:

1. **Symmetry**: `w[n] = w[windowSize - 1 - n]` (symmetric around center)
2. **Bounded**: All values between 0 and 1 (or close to it)
3. **Peak at center**: Maximum value typically at or near the middle
4. **Deterministic**: Same inputs always produce same outputs

### Window Function Comparison

| Window | Edge Values | Peak Value | Complexity | Main Use Case |
|--------|-------------|------------|------------|---------------|
| Rectangular | 1.0 | 1.0 | Simplest | Transient signals, baseline |
| Triangular | 0.0 | 1.0 | Simple | Basic spectral leakage reduction |
| Hanning | 0.0 | 1.0 | Moderate | General-purpose spectral analysis |
| Hamming | ~0.08 | 1.0 | Moderate | Speech/audio processing |
| Bartlett | 0.0 | ~1.0 | Simple | Power spectrum estimation |
| BartlettHann | 0.0 | 1.0 | Moderate | Balanced spectral characteristics |
| Cosine | 0.0 | 1.0 | Simple | Smooth transitions |
| Welch | 0.0 | 1.0 | Simple | Welch's power spectrum method |

---

## Current Implementation Status

### What's Already Done

All eight window functions are **fully implemented** in `/c/Users/cheat/source/repos/AiDotNet/src/WindowFunctions/`:

```
RectangularWindow.cs    - ✅ Implemented
TriangularWindow.cs     - ✅ Implemented
HanningWindow.cs        - ✅ Implemented
HammingWindow.cs        - ✅ Implemented
BartlettWindow.cs       - ✅ Implemented
BartlettHannWindow.cs   - ✅ Implemented
CosineWindow.cs         - ✅ Implemented
WelchWindow.cs          - ✅ Implemented
```

### What's Missing

**TEST COVERAGE**: There are **NO** unit tests for any of these window functions.

### Implementation Pattern

All window functions follow this pattern:

```csharp
public class SomeWindow<T> : IWindowFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public SomeWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public Vector<T> Create(int windowSize)
    {
        // Create vector and populate with window function values
        Vector<T> window = new Vector<T>(windowSize);
        for (int n = 0; n < windowSize; n++)
        {
            // Calculate window value at position n
            window[n] = /* formula */;
        }
        return window;
    }

    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.SomeWindow;
}
```

### Where Tests Should Go

Tests should be created in:
```
/c/Users/cheat/source/repos/AiDotNet/tests/UnitTests/WindowFunctions/
```

This directory **does not exist yet** and needs to be created.

---

## Testing Requirements

### What to Test

For each window function, you need to test:

#### 1. Basic Functionality
- ✅ Window can be created with valid sizes
- ✅ Returned vector has correct length
- ✅ Values are in expected range

#### 2. Mathematical Correctness
- ✅ Formula produces expected values at key positions (edges, center, quarters)
- ✅ Values match published reference values (MATLAB, NumPy, etc.)
- ✅ Symmetry property holds: `w[n] = w[windowSize - 1 - n]`

#### 3. Edge Cases
- ✅ Size = 1 (single-point window)
- ✅ Size = 2 (two-point window)
- ✅ Large sizes (e.g., 1024, 4096)
- ✅ Odd vs even sizes (some formulas differ)

#### 4. Type Safety
- ✅ Works with `float` type
- ✅ Works with `double` type
- ✅ Generic type operations are correct

#### 5. Window Properties
- ✅ **Normalization**: Peak value is at expected level (usually 1.0)
- ✅ **DC Gain**: Sum of window values (energy preservation)
- ✅ **Symmetry**: Values mirror around center
- ✅ **Coherent Gain**: Average value of window (for amplitude correction)

#### 6. Integration
- ✅ `GetWindowFunctionType()` returns correct enum value
- ✅ Window works in typical signal processing pipeline
- ✅ No unexpected exceptions or edge behavior

### Test Organization

Create one test class per window function:

```
WindowFunctions/
├── RectangularWindowTests.cs
├── TriangularWindowTests.cs
├── HanningWindowTests.cs
├── HammingWindowTests.cs
├── BartlettWindowTests.cs
├── BartlettHannWindowTests.cs
├── CosineWindowTests.cs
└── WelchWindowTests.cs
```

Each test class should have 15-25 test methods covering all aspects above.

---

## Mathematical Background

### 1. Rectangular Window

**Formula**: `w[n] = 1` for all `n`

**Properties**:
- Simplest window (no modification)
- Maximum energy preservation
- Worst spectral leakage
- Edge values: 1.0
- Peak value: 1.0
- Sum (size N): N

**Key Test Values** (size 8):
```
n:     0    1    2    3    4    5    6    7
w[n]:  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
```

### 2. Triangular Window

**Formula**: `w[n] = 1 - |2n - L| / L` where `L = windowSize - 1`

**Properties**:
- Linear increase then decrease
- Reaches 0 at edges (for size ≥ 3)
- Simple implementation
- Edge values: 0.0 (size ≥ 3)
- Peak value: 1.0 (at center)

**Key Test Values** (size 7):
```
n:     0      1      2      3      4      5      6
w[n]:  0.0    0.333  0.667  1.0    0.667  0.333  0.0
```

**Key Test Values** (size 8):
```
n:     0      1      2      3      4      5      6      7
w[n]:  0.0    0.286  0.571  0.857  0.857  0.571  0.286  0.0
```

### 3. Hanning Window

**Formula**: `w[n] = 0.5 * (1 - cos(2πn / (N - 1)))`

**Properties**:
- Raised cosine (single term)
- Reaches exactly 0 at both edges
- Good general-purpose window
- Edge values: 0.0
- Peak value: 1.0 (at center)

**Key Test Values** (size 8):
```
n:     0      1      2      3      4      5      6      7
w[n]:  0.0    0.188  0.611  0.950  0.950  0.611  0.188  0.0
```

### 4. Hamming Window

**Formula**: `w[n] = 0.54 - 0.46 * cos(2πn / (N - 1))`

**Alternative**: `w[n] = α - β * cos(2πn / (N - 1))` where α = 0.54, β = 0.46

**Properties**:
- Optimized raised cosine
- Does NOT reach 0 at edges (~0.08)
- Minimizes first sidelobe
- Edge values: ~0.08
- Peak value: 1.0 (at center)

**Key Test Values** (size 8):
```
n:     0      1      2      3      4      5      6      7
w[n]:  0.08   0.253  0.642  0.954  0.954  0.642  0.253  0.08
```

### 5. Bartlett Window

**Formula**:
```
For even N:
  w[n] = 2n/(N-1)           for n ≤ (N-1)/2
  w[n] = 2 - 2n/(N-1)       for n > (N-1)/2

For odd N: Similar but slightly different
```

**Properties**:
- Triangular, reaches 0 at edges
- Used in Bartlett's method for power spectral density
- Edge values: 0.0
- Peak value: 1.0 (at center)

**Key Test Values** (size 8):
```
n:     0      1      2      3      4      5      6      7
w[n]:  0.0    0.286  0.571  0.857  0.857  0.571  0.286  0.0
```

### 6. Bartlett-Hann Window

**Formula**: `w[n] = 0.62 - 0.48 * |n/(N-1) - 0.5| - 0.38 * cos(2πn/(N-1))`

**Properties**:
- Hybrid of Bartlett and Hann
- Reaches 0 at edges
- Better sidelobe performance than Bartlett
- Edge values: 0.0
- Peak value: 1.0 (at center)

**Key Test Values** (size 8):
```
n:     0      1      2      3      4      5      6      7
w[n]:  0.0    0.270  0.726  0.995  0.995  0.726  0.270  0.0
```

### 7. Cosine Window

**Formula**: `w[n] = sin(πn / (N - 1))`

**Alternative**: `w[n] = cos(π(n/(N-1) - 0.5))`

**Properties**:
- Simple sine-based window
- Reaches 0 at edges
- Smooth shape
- Edge values: 0.0
- Peak value: 1.0 (at center)

**Key Test Values** (size 8):
```
n:     0      1      2      3      4      5      6      7
w[n]:  0.0    0.434  0.782  0.975  0.975  0.782  0.434  0.0
```

### 8. Welch Window

**Formula**: `w[n] = 1 - ((n - (N-1)/2) / ((N+1)/2))²`

**Properties**:
- Parabolic (quadratic) shape
- Reaches 0 at edges
- Used in Welch's method
- Edge values: 0.0
- Peak value: 1.0 (at center)

**Key Test Values** (size 8):
```
n:     0      1      2      3      4      5      6      7
w[n]:  0.0    0.438  0.750  0.938  0.938  0.750  0.438  0.0
```

---

## Step-by-Step Testing Guide

### Step 1: Set Up Test Project Structure

#### 1.1 Create Test Directory

```bash
cd /c/Users/cheat/source/repos/AiDotNet/tests/UnitTests
mkdir WindowFunctions
```

#### 1.2 Verify Test Project Configuration

The test project should already have:
- Reference to main AiDotNet project
- xUnit test framework
- Test runner integration

Check `/c/Users/cheat/source/repos/AiDotNet/tests/AiDotNetTests.csproj`:
```xml
<ItemGroup>
  <PackageReference Include="xunit" Version="..." />
  <PackageReference Include="xunit.runner.visualstudio" Version="..." />
</ItemGroup>
```

### Step 2: Create Test Class Template

#### 2.1 Basic Test Class Structure

Create `/c/Users/cheat/source/repos/AiDotNet/tests/UnitTests/WindowFunctions/RectangularWindowTests.cs`:

```csharp
using System;
using AiDotNet.WindowFunctions;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.WindowFunctions
{
    public class RectangularWindowTests
    {
        [Fact]
        public void Constructor_CreatesValidWindow()
        {
            // Arrange & Act
            var window = new RectangularWindow<double>();

            // Assert
            Assert.NotNull(window);
        }

        [Fact]
        public void Create_WithValidSize_ReturnsCorrectLength()
        {
            // Arrange
            var window = new RectangularWindow<double>();
            var size = 16;

            // Act
            var result = window.Create(size);

            // Assert
            Assert.Equal(size, result.Length);
        }

        [Fact]
        public void GetWindowFunctionType_ReturnsRectangular()
        {
            // Arrange
            var window = new RectangularWindow<double>();

            // Act
            var type = window.GetWindowFunctionType();

            // Assert
            Assert.Equal(WindowFunctionType.Rectangular, type);
        }

        // Add more tests...
    }
}
```

### Step 3: Test Mathematical Correctness

#### 3.1 Test Formula at Key Positions

For **Rectangular Window** (simplest):

```csharp
[Fact]
public void Create_AllValuesAreOne()
{
    // Arrange
    var window = new RectangularWindow<double>();
    var size = 8;

    // Act
    var result = window.Create(size);

    // Assert
    for (int i = 0; i < size; i++)
    {
        Assert.Equal(1.0, result[i], 10);  // 10 decimal places precision
    }
}
```

For **Hanning Window** (more complex):

```csharp
[Fact]
public void Create_Size8_ProducesExpectedValues()
{
    // Arrange
    var window = new HanningWindow<double>();
    var size = 8;

    // Expected values from reference (MATLAB/NumPy)
    var expected = new double[]
    {
        0.0,
        0.1882550990706332,
        0.6112604669781572,
        0.9504844339512095,
        0.9504844339512095,
        0.6112604669781572,
        0.1882550990706332,
        0.0
    };

    // Act
    var result = window.Create(size);

    // Assert
    for (int i = 0; i < size; i++)
    {
        Assert.Equal(expected[i], result[i], 10);
    }
}
```

#### 3.2 Test Edge Values

```csharp
[Fact]
public void Create_EdgeValuesAreZero()  // For windows that reach zero
{
    // Arrange
    var window = new HanningWindow<double>();
    var size = 16;

    // Act
    var result = window.Create(size);

    // Assert
    Assert.Equal(0.0, result[0], 10);
    Assert.Equal(0.0, result[size - 1], 10);
}

[Fact]
public void Create_EdgeValuesAreNonZero()  // For Hamming window
{
    // Arrange
    var window = new HammingWindow<double>();
    var size = 16;

    // Act
    var result = window.Create(size);

    // Assert
    Assert.True(result[0] > 0.07 && result[0] < 0.09);  // ~0.08
    Assert.True(result[size - 1] > 0.07 && result[size - 1] < 0.09);
}
```

#### 3.3 Test Peak Value

```csharp
[Fact]
public void Create_PeakValueIsOne()
{
    // Arrange
    var window = new HanningWindow<double>();
    var size = 16;

    // Act
    var result = window.Create(size);

    // Assert
    var maxValue = result.Max();
    Assert.True(maxValue >= 0.99 && maxValue <= 1.01);
}
```

### Step 4: Test Symmetry

#### 4.1 Symmetry Test (Critical Property)

All windows should be symmetric around the center:

```csharp
[Fact]
public void Create_IsSymmetric()
{
    // Arrange
    var window = new HanningWindow<double>();
    var size = 16;

    // Act
    var result = window.Create(size);

    // Assert
    for (int i = 0; i < size / 2; i++)
    {
        var leftValue = result[i];
        var rightValue = result[size - 1 - i];
        Assert.Equal(leftValue, rightValue, 10);
    }
}

[Fact]
public void Create_OddSize_IsSymmetric()
{
    // Arrange
    var window = new HanningWindow<double>();
    var size = 15;  // Odd size

    // Act
    var result = window.Create(size);

    // Assert
    for (int i = 0; i < size / 2; i++)
    {
        var leftValue = result[i];
        var rightValue = result[size - 1 - i];
        Assert.Equal(leftValue, rightValue, 10);
    }
}
```

### Step 5: Test Edge Cases

#### 5.1 Single-Point Window

```csharp
[Fact]
public void Create_Size1_ReturnsValidWindow()
{
    // Arrange
    var window = new HanningWindow<double>();

    // Act
    var result = window.Create(1);

    // Assert
    Assert.Equal(1, result.Length);
    // For Hanning: w[0] = 0.5 * (1 - cos(0)) = 0 when N=1
    // But some implementations may handle this specially
    Assert.True(result[0] >= 0.0 && result[0] <= 1.0);
}
```

#### 5.2 Two-Point Window

```csharp
[Fact]
public void Create_Size2_ReturnsValidWindow()
{
    // Arrange
    var window = new HanningWindow<double>();

    // Act
    var result = window.Create(2);

    // Assert
    Assert.Equal(2, result.Length);
    Assert.Equal(0.0, result[0], 10);  // First edge
    Assert.Equal(0.0, result[1], 10);  // Second edge
}
```

#### 5.3 Large Window

```csharp
[Fact]
public void Create_LargeSize_CompletesWithoutError()
{
    // Arrange
    var window = new HanningWindow<double>();
    var size = 4096;

    // Act
    var result = window.Create(size);

    // Assert
    Assert.Equal(size, result.Length);
    Assert.True(result.Max() <= 1.0);
    Assert.True(result.Min() >= 0.0);
}
```

### Step 6: Test Type Safety

#### 6.1 Float Type

```csharp
[Fact]
public void Create_FloatType_WorksCorrectly()
{
    // Arrange
    var window = new HanningWindow<float>();
    var size = 8;

    // Act
    var result = window.Create(size);

    // Assert
    Assert.Equal(size, result.Length);
    Assert.Equal(0.0f, result[0], 5);  // 5 decimal places for float
    Assert.Equal(0.0f, result[size - 1], 5);
}
```

#### 6.2 Double Type

```csharp
[Fact]
public void Create_DoubleType_WorksCorrectly()
{
    // Arrange
    var window = new HanningWindow<double>();
    var size = 8;

    // Act
    var result = window.Create(size);

    // Assert
    Assert.Equal(size, result.Length);
    Assert.Equal(0.0, result[0], 10);  // 10 decimal places for double
    Assert.Equal(0.0, result[size - 1], 10);
}
```

### Step 7: Test Window Properties

#### 7.1 DC Gain (Sum of Window)

```csharp
[Fact]
public void Create_DCGain_IsReasonable()
{
    // Arrange
    var window = new HanningWindow<double>();
    var size = 16;

    // Act
    var result = window.Create(size);
    double sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += result[i];
    }

    // Assert
    // For Hanning window, DC gain ≈ N/2
    var expectedGain = size / 2.0;
    Assert.True(Math.Abs(sum - expectedGain) < 0.1);
}
```

#### 7.2 Coherent Gain (Average Value)

```csharp
[Fact]
public void Create_CoherentGain_IsReasonable()
{
    // Arrange
    var window = new HanningWindow<double>();
    var size = 16;

    // Act
    var result = window.Create(size);
    double sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += result[i];
    }
    var coherentGain = sum / size;

    // Assert
    // For Hanning window, coherent gain ≈ 0.5
    Assert.True(coherentGain > 0.45 && coherentGain < 0.55);
}
```

### Step 8: Create Reference Values

#### 8.1 Generate Reference Values with Python/NumPy

Use this Python script to generate reference values for testing:

```python
import numpy as np

# Rectangular
size = 8
window = np.ones(size)
print("Rectangular:", window)

# Hanning
window = np.hanning(size)
print("Hanning:", window)

# Hamming
window = np.hamming(size)
print("Hamming:", window)

# Bartlett
window = np.bartlett(size)
print("Bartlett:", window)

# Bartlett-Hann
window = np.bartlett_hann(size)  # NumPy function name
print("Bartlett-Hann:", window)
```

#### 8.2 Or Use MATLAB Reference

```matlab
% MATLAB reference
size = 8;

rect = rectwin(size);
tri = triang(size);
hann = hanning(size);
hamm = hamming(size);
bart = bartlett(size);
barthann = barthannwin(size);
cos = cosine(size);
welch = welchwin(size);
```

### Step 9: Test Comparison with Known Windows

#### 9.1 Rectangular vs Other Windows

```csharp
[Fact]
public void Create_HanningPreservesLessEnergyThanRectangular()
{
    // Arrange
    var rectangular = new RectangularWindow<double>();
    var hanning = new HanningWindow<double>();
    var size = 16;

    // Act
    var rectResult = rectangular.Create(size);
    var hannResult = hanning.Create(size);

    double rectEnergy = 0;
    double hannEnergy = 0;
    for (int i = 0; i < size; i++)
    {
        rectEnergy += rectResult[i] * rectResult[i];
        hannEnergy += hannResult[i] * hannResult[i];
    }

    // Assert
    // Hanning should preserve less energy due to tapering
    Assert.True(hannEnergy < rectEnergy);
}
```

### Step 10: Run and Verify Tests

#### 10.1 Build the Test Project

```bash
cd /c/Users/cheat/source/repos/AiDotNet
dotnet build tests/AiDotNetTests.csproj
```

#### 10.2 Run Tests

```bash
dotnet test tests/AiDotNetTests.csproj --filter "FullyQualifiedName~WindowFunctions"
```

#### 10.3 Check Coverage

```bash
dotnet test tests/AiDotNetTests.csproj \
  /p:CollectCoverage=true \
  /p:CoverletOutputFormat=opencover \
  --filter "FullyQualifiedName~WindowFunctions"
```

---

## Common Pitfalls and How to Avoid Them

### Pitfall 1: Incorrect Precision in Assertions

**Problem**: Using `Assert.Equal(expected, actual)` for floating-point comparisons without specifying precision.

**Why it fails**: Floating-point arithmetic has rounding errors. `0.9504844339512095` might be stored as `0.9504844339512094`.

**Solution**: Always specify precision:

```csharp
// ❌ WRONG
Assert.Equal(0.9504844339512095, result[3]);

// ✅ CORRECT
Assert.Equal(0.9504844339512095, result[3], 10);  // 10 decimal places
```

### Pitfall 2: Not Testing Odd vs Even Sizes

**Problem**: Some window formulas behave differently for odd vs even sizes.

**Why it matters**: The center point calculation differs:
- Even size: center between two points
- Odd size: center exactly on one point

**Solution**: Test both:

```csharp
[Fact]
public void Create_EvenSize_IsSymmetric() { /* test size 16 */ }

[Fact]
public void Create_OddSize_IsSymmetric() { /* test size 15 */ }
```

### Pitfall 3: Not Testing Edge Cases

**Problem**: Only testing "normal" window sizes (e.g., 16, 32, 64).

**Why it fails**: Size 1 and size 2 windows can have division-by-zero or special-case logic.

**Solution**: Always test:
- Size 1 (degenerate case)
- Size 2 (minimal window)
- Size 3 (smallest useful odd size)
- Large size (e.g., 4096)

### Pitfall 4: Incorrect Reference Values

**Problem**: Using hand-calculated values instead of verified references.

**Why it fails**: Easy to make arithmetic mistakes.

**Solution**: Use values from:
- NumPy: `np.hanning(8)`
- MATLAB: `hanning(8)`
- Wolfram Alpha: verified calculations
- Published literature with tables

### Pitfall 5: Not Testing Symmetry Property

**Problem**: Assuming symmetry without testing it.

**Why it matters**: Symmetry is a fundamental property of all these windows. If broken, the window is incorrect.

**Solution**: Always include symmetry test:

```csharp
[Fact]
public void Create_IsSymmetric()
{
    var window = new HanningWindow<double>();
    var result = window.Create(16);

    for (int i = 0; i < 8; i++)
    {
        Assert.Equal(result[i], result[15 - i], 10);
    }
}
```

### Pitfall 6: Forgetting Float vs Double Differences

**Problem**: Only testing with `double` type.

**Why it fails**: Float has lower precision (7 digits vs 15-16 for double). Tests might pass for double but fail for float.

**Solution**: Test both types with appropriate precision:

```csharp
// Double: 10 decimal places
Assert.Equal(0.9504844339512095, resultDouble[3], 10);

// Float: 5 decimal places
Assert.Equal(0.9504844f, resultFloat[3], 5);
```

### Pitfall 7: Not Verifying Window Properties

**Problem**: Only testing formula correctness without verifying properties like DC gain, coherent gain.

**Why it matters**: These properties are used in signal processing calculations. Incorrect properties mean incorrect signal analysis.

**Solution**: Test derived properties:
- DC gain (sum of window)
- Coherent gain (average value)
- Energy (sum of squares)
- Peak-to-average ratio

### Pitfall 8: Using Magic Numbers

**Problem**: Hard-coding values without explanation.

**Why it fails**: Makes tests hard to understand and maintain.

**Solution**: Use comments and named constants:

```csharp
// ❌ WRONG
Assert.True(result[0] > 0.07 && result[0] < 0.09);

// ✅ CORRECT
// Hamming window has non-zero edge values ~0.08
const double HammingEdgeValueLow = 0.07;
const double HammingEdgeValueHigh = 0.09;
Assert.True(result[0] > HammingEdgeValueLow && result[0] < HammingEdgeValueHigh);
```

---

## Verification Checklist

Before submitting your implementation, verify:

### For Each Window Function (8 total):

- [ ] Test class created in `tests/UnitTests/WindowFunctions/`
- [ ] Test class follows naming convention: `{WindowName}WindowTests.cs`
- [ ] Namespace is `AiDotNetTests.UnitTests.WindowFunctions`

### Basic Tests (per window):

- [ ] Constructor creates valid window
- [ ] Create returns vector of correct length
- [ ] GetWindowFunctionType returns correct enum value

### Mathematical Correctness (per window):

- [ ] Formula produces expected values (verified against NumPy/MATLAB)
- [ ] Edge values are correct (0.0 or other expected value)
- [ ] Peak value is correct (usually 1.0)
- [ ] Center value is correct

### Symmetry Tests (per window):

- [ ] Even-sized windows are symmetric
- [ ] Odd-sized windows are symmetric
- [ ] Symmetry holds for various sizes (8, 15, 16, 31, 32)

### Edge Case Tests (per window):

- [ ] Size 1 window works correctly
- [ ] Size 2 window works correctly
- [ ] Large size (e.g., 4096) works correctly

### Type Safety Tests (per window):

- [ ] Works with `float` type
- [ ] Works with `double` type

### Window Properties Tests (per window):

- [ ] DC gain is reasonable
- [ ] Coherent gain is reasonable
- [ ] All values in range [0, 1] (or known range)

### Integration Tests:

- [ ] All tests pass with `dotnet test`
- [ ] No warnings in test output
- [ ] Code coverage shows all window Create methods covered
- [ ] Tests run in reasonable time (<1s per test class)

### Documentation:

- [ ] Test methods have descriptive names
- [ ] Complex tests have comments explaining expected behavior
- [ ] Reference values cite source (NumPy, MATLAB, etc.)

### Total Test Count:

Each window should have approximately **15-20 tests**, resulting in:
- **120-160 total tests** for all 8 windows

This comprehensive test suite ensures:
- ✅ Mathematical correctness
- ✅ Edge case handling
- ✅ Type safety
- ✅ Window properties
- ✅ Integration with AiDotNet

---

## Example Complete Test Class

Here's a complete example for **HanningWindow**:

```csharp
using System;
using AiDotNet.WindowFunctions;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.WindowFunctions
{
    public class HanningWindowTests
    {
        [Fact]
        public void Constructor_CreatesValidWindow()
        {
            // Arrange & Act
            var window = new HanningWindow<double>();

            // Assert
            Assert.NotNull(window);
        }

        [Fact]
        public void Create_WithValidSize_ReturnsCorrectLength()
        {
            // Arrange
            var window = new HanningWindow<double>();
            var size = 16;

            // Act
            var result = window.Create(size);

            // Assert
            Assert.Equal(size, result.Length);
        }

        [Fact]
        public void GetWindowFunctionType_ReturnsHanning()
        {
            // Arrange
            var window = new HanningWindow<double>();

            // Act
            var type = window.GetWindowFunctionType();

            // Assert
            Assert.Equal(WindowFunctionType.Hanning, type);
        }

        [Fact]
        public void Create_Size8_ProducesExpectedValues()
        {
            // Arrange
            var window = new HanningWindow<double>();
            var size = 8;

            // Expected values from NumPy: np.hanning(8)
            var expected = new double[]
            {
                0.0,
                0.1882550990706332,
                0.6112604669781572,
                0.9504844339512095,
                0.9504844339512095,
                0.6112604669781572,
                0.1882550990706332,
                0.0
            };

            // Act
            var result = window.Create(size);

            // Assert
            for (int i = 0; i < size; i++)
            {
                Assert.Equal(expected[i], result[i], 10);
            }
        }

        [Fact]
        public void Create_EdgeValuesAreZero()
        {
            // Arrange
            var window = new HanningWindow<double>();
            var size = 16;

            // Act
            var result = window.Create(size);

            // Assert
            Assert.Equal(0.0, result[0], 10);
            Assert.Equal(0.0, result[size - 1], 10);
        }

        [Fact]
        public void Create_PeakValueIsOne()
        {
            // Arrange
            var window = new HanningWindow<double>();
            var size = 16;

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
            var window = new HanningWindow<double>();
            var size = 16;

            // Act
            var result = window.Create(size);

            // Assert
            for (int i = 0; i < size / 2; i++)
            {
                Assert.Equal(result[i], result[size - 1 - i], 10);
            }
        }

        [Fact]
        public void Create_IsSymmetric_OddSize()
        {
            // Arrange
            var window = new HanningWindow<double>();
            var size = 15;

            // Act
            var result = window.Create(size);

            // Assert
            for (int i = 0; i < size / 2; i++)
            {
                Assert.Equal(result[i], result[size - 1 - i], 10);
            }
        }

        [Fact]
        public void Create_Size1_ReturnsValidWindow()
        {
            // Arrange
            var window = new HanningWindow<double>();

            // Act
            var result = window.Create(1);

            // Assert
            Assert.Equal(1, result.Length);
            Assert.True(result[0] >= 0.0 && result[0] <= 1.0);
        }

        [Fact]
        public void Create_Size2_ReturnsValidWindow()
        {
            // Arrange
            var window = new HanningWindow<double>();

            // Act
            var result = window.Create(2);

            // Assert
            Assert.Equal(2, result.Length);
            Assert.Equal(0.0, result[0], 10);
            Assert.Equal(0.0, result[1], 10);
        }

        [Fact]
        public void Create_LargeSize_CompletesWithoutError()
        {
            // Arrange
            var window = new HanningWindow<double>();
            var size = 4096;

            // Act
            var result = window.Create(size);

            // Assert
            Assert.Equal(size, result.Length);
            Assert.True(result.Max() <= 1.0);
            Assert.True(result.Min() >= 0.0);
        }

        [Fact]
        public void Create_FloatType_WorksCorrectly()
        {
            // Arrange
            var window = new HanningWindow<float>();
            var size = 8;

            // Act
            var result = window.Create(size);

            // Assert
            Assert.Equal(size, result.Length);
            Assert.Equal(0.0f, result[0], 5);
            Assert.Equal(0.0f, result[size - 1], 5);
        }

        [Fact]
        public void Create_DoubleType_WorksCorrectly()
        {
            // Arrange
            var window = new HanningWindow<double>();
            var size = 8;

            // Act
            var result = window.Create(size);

            // Assert
            Assert.Equal(size, result.Length);
            Assert.Equal(0.0, result[0], 10);
            Assert.Equal(0.0, result[size - 1], 10);
        }

        [Fact]
        public void Create_DCGain_IsApproximatelyHalfOfSize()
        {
            // Arrange
            var window = new HanningWindow<double>();
            var size = 16;

            // Act
            var result = window.Create(size);
            double sum = 0;
            for (int i = 0; i < size; i++)
            {
                sum += result[i];
            }

            // Assert
            // Hanning window has DC gain ≈ N/2
            var expectedGain = size / 2.0;
            Assert.True(Math.Abs(sum - expectedGain) < 0.1);
        }

        [Fact]
        public void Create_CoherentGain_IsApproximatelyPointFive()
        {
            // Arrange
            var window = new HanningWindow<double>();
            var size = 16;

            // Act
            var result = window.Create(size);
            double sum = 0;
            for (int i = 0; i < size; i++)
            {
                sum += result[i];
            }
            var coherentGain = sum / size;

            // Assert
            // Hanning window has coherent gain ≈ 0.5
            Assert.True(coherentGain > 0.45 && coherentGain < 0.55);
        }

        [Fact]
        public void Create_AllValuesInRangeZeroToOne()
        {
            // Arrange
            var window = new HanningWindow<double>();
            var size = 32;

            // Act
            var result = window.Create(size);

            // Assert
            for (int i = 0; i < size; i++)
            {
                Assert.True(result[i] >= 0.0 && result[i] <= 1.0);
            }
        }

        [Fact]
        public void Create_PreservesLessEnergyThanRectangular()
        {
            // Arrange
            var hanning = new HanningWindow<double>();
            var rectangular = new RectangularWindow<double>();
            var size = 16;

            // Act
            var hannResult = hanning.Create(size);
            var rectResult = rectangular.Create(size);

            double hannEnergy = 0;
            double rectEnergy = 0;
            for (int i = 0; i < size; i++)
            {
                hannEnergy += hannResult[i] * hannResult[i];
                rectEnergy += rectResult[i] * rectResult[i];
            }

            // Assert
            // Hanning should preserve less energy due to tapering
            Assert.True(hannEnergy < rectEnergy);
        }
    }
}
```

This test class provides comprehensive coverage of the Hanning window, including:
- ✅ 17 test methods
- ✅ All critical properties tested
- ✅ Edge cases covered
- ✅ Type safety verified
- ✅ Reference values from NumPy
- ✅ Clear, descriptive test names
- ✅ Appropriate precision in assertions

Replicate this pattern for all 8 window functions in Issue #366.

---

## Next Steps

1. Create test directory: `tests/UnitTests/WindowFunctions/`
2. Implement test classes for all 8 windows (use template above)
3. Generate reference values using NumPy/MATLAB
4. Run tests and verify 100% pass rate
5. Check code coverage (should be >95% for window implementations)
6. Document any unexpected behaviors or edge cases discovered
7. Submit PR with all tests

**Estimated effort**: 8-12 hours for all 8 window functions (1-1.5 hours per window)

Good luck!
