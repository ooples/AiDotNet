# Issue #353: Junior Developer Implementation Guide
## Spline Interpolation - Unit Tests and Validation

---

## Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [What are Splines?](#what-are-splines)
3. [Understanding Spline Interpolation Methods](#understanding-spline-interpolation-methods)
4. [Current Implementation Status](#current-implementation-status)
5. [Testing Strategy](#testing-strategy)
6. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
7. [Mathematical Background](#mathematical-background)
8. [Common Pitfalls](#common-pitfalls)

---

## Understanding the Problem

### What Are We Solving?

The AiDotNet library has **nine spline interpolation implementations** that currently have **0% test coverage**. We need to create comprehensive unit tests that verify numerical accuracy, smoothness properties, monotonicity preservation, and boundary conditions.

### The Core Issue

**Currently**: The spline methods exist but lack validation:
- No tests verify smoothness (continuous derivatives)
- No tests check monotonicity preservation
- No tests validate numerical accuracy
- Developers cannot trust these methods for production use

**Goal**: Create comprehensive test suites that:
1. Verify numerical accuracy against known functions
2. Test smoothness properties (C1, C2 continuity)
3. Test monotonicity preservation (where applicable)
4. Test boundary conditions and edge cases
5. Provide confidence for production use

### Methods to Test (Issue #353)

1. **CubicSplineInterpolation** (Natural Spline) - Second derivative = 0 at boundaries
2. **CubicBSplineInterpolation** - Uniform B-spline basis
3. **CatmullRomSplineInterpolation** - Tangent-based spline
4. **HermiteInterpolation** - Uses point values and derivatives
5. **AkimaInterpolation** - Avoids overshoot
6. **MonotoneCubicInterpolation** (PCHIP) - Preserves monotonicity
7. **ClampedSplineInterpolation** - Specified derivatives at boundaries
8. **AdaptiveCubicSplineInterpolation** - Varies smoothness based on data
9. **KochanekBartelsSplineInterpolation** - Tension, continuity, bias parameters

---

## What are Splines?

### Real-World Analogy

Imagine you're designing a roller coaster track. You have control points where the track must pass through, but you want the track to be smooth (no sharp corners that would hurt riders). Splines are mathematical curves that:

1. Pass through all your control points
2. Are smooth (no sudden changes in direction)
3. Can be customized for different "feel" (gentle curves vs. steep drops)

### In AI and Data Processing

Splines are used for:
- **Smooth Curve Fitting**: Creating smooth paths through data points
- **Animation**: Smooth motion paths in computer graphics
- **Signal Processing**: Smoothing noisy sensor data
- **Image Processing**: Smooth edges and contours
- **Time Series**: Interpolating missing values in temporal data

### Why Splines Over Linear Interpolation?

**Linear**: Sharp corners at each data point
```
    *
   /
  /   *
 /   /
*   /
   /
  *
```

**Spline**: Smooth curves through points
```
    *
  /
 /    *
/    /
*  /
 /
*
```

---

## Understanding Spline Interpolation Methods

### 1. Cubic Spline (Natural Spline)

**What it does**: Creates piecewise cubic polynomials with continuous first and second derivatives.

**Properties**:
- C2 continuous (smooth curves, smooth slope changes)
- Natural boundary conditions: second derivative = 0 at endpoints
- Minimizes curvature (smoothest possible curve)
- Requires at least 3 points

**Mathematical Property**:
```
For each interval [x_i, x_{i+1}]:
  S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3

Continuity conditions:
  S_i(x_{i+1}) = S_{i+1}(x_{i+1})           [C0: function continuous]
  S'_i(x_{i+1}) = S'_{i+1}(x_{i+1})         [C1: first derivative continuous]
  S''_i(x_{i+1}) = S''_{i+1}(x_{i+1})       [C2: second derivative continuous]

Boundary conditions (natural spline):
  S''_0(x_0) = 0
  S''_{n-1}(x_n) = 0
```

**Use Cases**:
- General-purpose smooth interpolation
- When you want the smoothest possible curve
- Scientific data visualization

**Test Focus**:
- Verify passes through all data points
- Verify C2 continuity (second derivative continuous)
- Verify natural boundary conditions
- Test against known smooth functions

### 2. Cubic B-Spline

**What it does**: Uses basis functions to create smooth curves (may not pass through all control points).

**Properties**:
- C2 continuous everywhere
- Local control (moving one point affects only nearby curve)
- Approximating (doesn't necessarily pass through points)
- Requires at least 4 points

**Difference from Cubic Spline**:
- **Cubic Spline**: Passes through all data points (interpolating)
- **B-Spline**: Approximates data points (may not pass through)

**Use Cases**:
- Computer graphics (smooth curves)
- When local control is important
- When exact passage through points isn't required

**Test Focus**:
- Verify smoothness (C2 continuity)
- Verify local control property
- Test curve stays near data points
- Test uniform knot vector handling

### 3. Catmull-Rom Spline

**What it does**: Creates smooth curves using tangent vectors derived from neighboring points.

**Properties**:
- C1 continuous (smooth curve, smooth slope)
- Passes through all interior data points
- Tangent at each point derived from neighbors
- Requires at least 4 points

**Tangent Formula**:
```
Tangent at point i:
  m_i = (p_{i+1} - p_{i-1}) / (x_{i+1} - x_{i-1})
```

**Use Cases**:
- Camera paths in animation
- Smooth motion in games
- When you want control over tangents

**Test Focus**:
- Verify passes through data points
- Verify C1 continuity
- Test tangent calculation
- Test with equally-spaced points

### 4. Hermite Interpolation

**What it does**: Interpolates using both function values and derivative values.

**Properties**:
- C1 continuous
- Requires both y-values and derivatives at each point
- Full control over tangents
- Requires at least 2 points + derivatives

**Input**:
- `x[i]`: x-coordinates
- `y[i]`: y-values
- `dy[i]`: derivatives at each point

**Use Cases**:
- When you know derivatives (e.g., velocity and position)
- Physics simulations
- When precise control over slopes is needed

**Test Focus**:
- Verify passes through all points
- Verify derivatives match at all points
- Test with known functions (where derivatives are known)

### 5. Akima Interpolation

**What it does**: Creates smooth curves that avoid overshoot and artificial oscillations.

**Properties**:
- C1 continuous
- Minimizes overshoot (no wild oscillations)
- Local behavior (changes in one area don't affect distant areas)
- Requires at least 5 points

**Key Advantage**: Handles sharp changes in data without creating waves.

**Visual Example**:
```
Data:  *-----*-----*   *---*---*
           ^
       Sharp change

Cubic Spline:  Creates oscillations around sharp change
Akima:         Smooth but follows data closely (no oscillations)
```

**Use Cases**:
- Financial data (stock prices with sudden changes)
- Sensor data with noise
- When you want to avoid overshooting

**Test Focus**:
- Verify no overshoot near sharp changes
- Verify local control
- Compare with cubic spline (Akima should have less oscillation)
- Test minimum 5-point requirement

### 6. Monotone Cubic (PCHIP - Piecewise Cubic Hermite Interpolating Polynomial)

**What it does**: Creates smooth curves that preserve monotonicity (if data goes up, curve goes up).

**Properties**:
- C1 continuous
- Preserves monotonicity
- No overshoot
- Requires at least 2 points

**Monotonicity**:
```
If y[i] <= y[i+1] <= y[i+2]:
  Then interpolated curve is also non-decreasing in that region
```

**Use Cases**:
- Cumulative data (never decreases)
- Probability distributions
- Economic indicators (GDP, etc.)
- Any data that should not reverse direction

**Test Focus**:
- **Critical**: Verify monotonicity preservation
- Test with strictly increasing data
- Test with strictly decreasing data
- Test with mixed monotone regions

### 7. Clamped Spline

**What it does**: Like natural spline, but you specify the derivatives at the boundaries.

**Properties**:
- C2 continuous
- Specified first derivatives at endpoints
- More control than natural spline
- Requires at least 3 points + 2 boundary derivatives

**Boundary Conditions**:
```
S'(x_0) = dy_0  (specified)
S'(x_n) = dy_n  (specified)

vs. Natural Spline:
S''(x_0) = 0
S''(x_n) = 0
```

**Use Cases**:
- When you know the slopes at boundaries
- Physics (known velocity at endpoints)
- Closed curves (start and end slopes match)

**Test Focus**:
- Verify specified derivatives at boundaries
- Test with various boundary conditions
- Compare with natural spline

### 8. Adaptive Cubic Spline

**What it does**: Varies the smoothness based on data characteristics.

**Properties**:
- Adapts to data (smooth in smooth regions, sharper in sharp regions)
- C1 or C2 continuous (depends on implementation)
- May use knot refinement

**Use Cases**:
- Data with mixed smooth and sharp regions
- Image edge detection
- Signal processing with varying characteristics

**Test Focus**:
- Verify adaptation behavior
- Test with mixed smooth/sharp data
- Verify smoothness in smooth regions
- Verify sharpness in sharp regions

### 9. Kochanek-Bartels Spline (TCB Spline)

**What it does**: Spline with three parameters controlling curve shape.

**Parameters**:
- **Tension (T)**: How tight or loose the curve is (-1 to 1)
  - T = 1: Very tight (almost linear)
  - T = 0: Normal smoothness
  - T = -1: Very loose (more curves)

- **Continuity (C)**: How sharp corners can be (-1 to 1)
  - C = 1: Sharp corners allowed
  - C = 0: Smooth transitions
  - C = -1: Very smooth transitions

- **Bias (B)**: Direction of overshoot (-1 to 1)
  - B = 1: Overshoot toward next point
  - B = 0: Balanced
  - B = -1: Overshoot toward previous point

**Use Cases**:
- Animation (fine control over motion)
- Graphics (artistic control)
- When you need precise control over curve shape

**Test Focus**:
- Test each parameter independently
- Test parameter combinations
- Verify effect of tension, continuity, bias
- Test default values (T=C=B=0)

---

## Current Implementation Status

### Existing Files

**Source Files** (all in `C:\Users\cheat\source\repos\AiDotNet\src\Interpolation\`):
1. `CubicSplineInterpolation.cs`
2. `CubicBSplineInterpolation.cs`
3. `CatmullRomSplineInterpolation.cs`
4. `HermiteInterpolation.cs`
5. `AkimaInterpolation.cs`
6. `MonotoneCubicInterpolation.cs`
7. `ClampedSplineInterpolation.cs`
8. `AdaptiveCubicSplineInterpolation.cs`
9. `KochanekBartelsSplineInterpolation.cs`

**Test Status**:
- **Current**: 0% coverage (no tests exist)
- **Target**: 80%+ coverage with smoothness and monotonicity verification

---

## Testing Strategy

### Test Categories

#### 1. Exact Point Tests

**Purpose**: Verify spline passes through all data points (for interpolating splines).

```csharp
[Theory]
[InlineData(1.0, 10.0)]
[InlineData(2.0, 20.0)]
[InlineData(3.0, 15.0)]
public void Interpolate_AtKnownPoint_ReturnsExactValue(double x, double expectedY)
{
    var xData = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
    var yData = new Vector<double>(new[] { 10.0, 20.0, 15.0, 25.0, 20.0 });
    var spline = new CubicSplineInterpolation<double>(xData, yData);

    double result = spline.Interpolate(x);

    Assert.Equal(expectedY, result, precision: 10);
}
```

#### 2. Smoothness Tests (Continuity)

**Purpose**: Verify derivatives are continuous (C1 or C2 continuity).

**C1 Continuity Test** (first derivative continuous):
```csharp
[Fact]
public void Interpolate_HasContinuousFirstDerivative()
{
    // Arrange
    var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
    var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
    var spline = new CubicSplineInterpolation<double>(x, y);

    // Act - Test derivative continuity at x=1.0 (data point)
    double h = 1e-6;

    // Left derivative: (f(1) - f(1-h)) / h
    double leftDeriv = (spline.Interpolate(1.0) - spline.Interpolate(1.0 - h)) / h;

    // Right derivative: (f(1+h) - f(1)) / h
    double rightDeriv = (spline.Interpolate(1.0 + h) - spline.Interpolate(1.0)) / h;

    // Assert - Should be approximately equal (C1 continuity)
    Assert.Equal(leftDeriv, rightDeriv, precision: 4);
}
```

**C2 Continuity Test** (second derivative continuous):
```csharp
[Fact]
public void Interpolate_HasContinuousSecondDerivative()
{
    // Similar to C1 but compute second derivatives
    // d²f/dx² ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
}
```

#### 3. Monotonicity Preservation Tests

**Purpose**: Verify monotone splines preserve increasing/decreasing behavior.

```csharp
[Fact]
public void Interpolate_WithIncreasingData_ProducesIncreasingCurve()
{
    // Arrange - Strictly increasing data
    var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
    var y = new Vector<double>(new[] { 10.0, 15.0, 18.0, 25.0, 30.0 });
    var spline = new MonotoneCubicInterpolation<double>(x, y);

    // Act - Sample many points
    for (double testX = 0.0; testX < 4.0; testX += 0.1)
    {
        double y1 = spline.Interpolate(testX);
        double y2 = spline.Interpolate(testX + 0.1);

        // Assert - y2 should be >= y1 (monotone increasing)
        Assert.True(y2 >= y1, $"Non-monotonic at x={testX}: y1={y1}, y2={y2}");
    }
}
```

#### 4. Known Function Tests

**Purpose**: Test against mathematical functions we can verify.

**Example** (Sine function):
```csharp
[Fact]
public void Interpolate_SineFunction_HighAccuracy()
{
    // Arrange - Sample sine function
    int n = 20;
    var x = new Vector<double>(n);
    var y = new Vector<double>(n);

    for (int i = 0; i < n; i++)
    {
        x[i] = i * Math.PI / (n - 1);  // 0 to π
        y[i] = Math.Sin(x[i]);
    }

    var spline = new CubicSplineInterpolation<double>(x, y);

    // Act - Test at intermediate points
    for (double testX = 0.0; testX <= Math.PI; testX += 0.1)
    {
        double interpolated = spline.Interpolate(testX);
        double expected = Math.Sin(testX);

        // Assert - Should be very close
        Assert.Equal(expected, interpolated, precision: 3);
    }
}
```

#### 5. Boundary Condition Tests

**Purpose**: Test specific boundary conditions (natural, clamped, etc.).

**Natural Spline** (second derivative = 0 at boundaries):
```csharp
[Fact]
public void NaturalSpline_HasZeroSecondDerivativeAtBoundaries()
{
    // Arrange
    var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
    var y = new Vector<double>(new[] { 0.0, 1.0, 0.5, 1.5 });
    var spline = new CubicSplineInterpolation<double>(x, y);

    // Act - Compute second derivative at boundaries
    // d²f/dx² ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
    double h = 1e-4;
    double x0 = x[0];
    double xn = x[x.Length - 1];

    double secondDerivAt0 = (spline.Interpolate(x0 + h) - 2 * spline.Interpolate(x0) + spline.Interpolate(x0 + h * 0.5)) / (h * h / 4);
    double secondDerivAtN = (spline.Interpolate(xn - h * 0.5) - 2 * spline.Interpolate(xn) + spline.Interpolate(xn - h)) / (h * h / 4);

    // Assert - Should be approximately zero
    Assert.Equal(0.0, secondDerivAt0, precision: 2);
    Assert.Equal(0.0, secondDerivAtN, precision: 2);
}
```

**Clamped Spline** (specified derivatives at boundaries):
```csharp
[Fact]
public void ClampedSpline_HasSpecifiedDerivativesAtBoundaries()
{
    // Arrange
    var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
    var y = new Vector<double>(new[] { 0.0, 1.0, 0.5, 1.5 });
    double leftDerivative = 2.0;   // Specified
    double rightDerivative = -1.0; // Specified

    var spline = new ClampedSplineInterpolation<double>(x, y, leftDerivative, rightDerivative);

    // Act - Compute derivatives at boundaries
    double h = 1e-6;
    double computedLeftDeriv = (spline.Interpolate(x[0] + h) - spline.Interpolate(x[0])) / h;
    double computedRightDeriv = (spline.Interpolate(x[x.Length - 1]) - spline.Interpolate(x[x.Length - 1] - h)) / h;

    // Assert
    Assert.Equal(leftDerivative, computedLeftDeriv, precision: 4);
    Assert.Equal(rightDerivative, computedRightDeriv, precision: 4);
}
```

#### 6. Overshoot Tests (Akima)

**Purpose**: Verify Akima avoids overshoot near sharp changes.

```csharp
[Fact]
public void Akima_AvoidOvershoot_NearSharpChange()
{
    // Arrange - Data with sharp change
    var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 2.1, 3.0, 4.0, 5.0 });
    var y = new Vector<double>(new[] { 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0 });

    var akima = new AkimaInterpolation<double>(x, y);

    // Act - Sample points near sharp change
    for (double testX = 1.5; testX <= 2.5; testX += 0.05)
    {
        double interpolated = akima.Interpolate(testX);

        // Assert - Should stay between min and max of surrounding points
        Assert.True(interpolated >= -0.5 && interpolated <= 5.5,
            $"Overshoot detected at x={testX}: y={interpolated}");
    }
}
```

#### 7. Parameter Effect Tests (Kochanek-Bartels)

**Purpose**: Test effect of tension, continuity, bias parameters.

```csharp
[Theory]
[InlineData(1.0, 0.0, 0.0)]   // High tension (tight)
[InlineData(-1.0, 0.0, 0.0)]  // Low tension (loose)
[InlineData(0.0, 1.0, 0.0)]   // High continuity (sharp)
[InlineData(0.0, -1.0, 0.0)]  // Low continuity (smooth)
[InlineData(0.0, 0.0, 1.0)]   // Positive bias
[InlineData(0.0, 0.0, -1.0)]  // Negative bias
public void KochanekBartels_ParameterEffect_ProducesDifferentCurves(double tension, double continuity, double bias)
{
    // Arrange
    var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
    var y = new Vector<double>(new[] { 0.0, 1.0, 0.5, 1.5, 1.0 });

    var spline = new KochanekBartelsSplineInterpolation<double>(x, y, tension, continuity, bias);

    // Act - Sample curve
    double result = spline.Interpolate(1.5);

    // Assert - Should be finite and reasonable
    Assert.True(double.IsFinite(result));
    Assert.True(result >= -2.0 && result <= 3.0);  // Sanity check
}
```

---

## Step-by-Step Implementation Guide

### Phase 1: Set Up Test Infrastructure (1 hour)

#### AC 1.1: Create Test Files

Create test files for each spline type:

```bash
cd C:/Users/cheat/source/repos/AiDotNet/tests/UnitTests/Interpolation

# Create all spline test files
touch CubicSplineInterpolationTests.cs
touch CubicBSplineInterpolationTests.cs
touch CatmullRomSplineInterpolationTests.cs
touch HermiteInterpolationTests.cs
touch AkimaInterpolationTests.cs
touch MonotoneCubicInterpolationTests.cs
touch ClampedSplineInterpolationTests.cs
touch AdaptiveCubicSplineInterpolationTests.cs
touch KochanekBartelsSplineInterpolationTests.cs
```

#### AC 1.2: Create Test Helper Class

**File**: `C:\Users\cheat\source\repos\AiDotNet\tests\UnitTests\Interpolation\SplineTestHelpers.cs`

```csharp
namespace AiDotNetTests.UnitTests.Interpolation;

using AiDotNet;

/// <summary>
/// Helper methods for spline interpolation tests.
/// </summary>
public static class SplineTestHelpers
{
    /// <summary>
    /// Computes numerical first derivative using central difference.
    /// </summary>
    public static double ComputeFirstDerivative<T>(
        AiDotNet.Interfaces.IInterpolation<T> interpolator,
        T x,
        double h = 1e-6)
        where T : struct
    {
        var numOps = AiDotNet.MathHelper.GetNumericOperations<T>();
        var hT = numOps.FromDouble(h);

        var xPlusH = numOps.Add(x, hT);
        var xMinusH = numOps.Subtract(x, hT);

        var fPlusH = interpolator.Interpolate(xPlusH);
        var fMinusH = interpolator.Interpolate(xMinusH);

        var numerator = numOps.Subtract(fPlusH, fMinusH);
        var denominator = numOps.FromDouble(2 * h);

        return Convert.ToDouble(numOps.Divide(numerator, denominator));
    }

    /// <summary>
    /// Computes numerical second derivative.
    /// </summary>
    public static double ComputeSecondDerivative<T>(
        AiDotNet.Interfaces.IInterpolation<T> interpolator,
        T x,
        double h = 1e-4)
        where T : struct
    {
        var numOps = AiDotNet.MathHelper.GetNumericOperations<T>();
        var hT = numOps.FromDouble(h);

        var xPlusH = numOps.Add(x, hT);
        var xMinusH = numOps.Subtract(x, hT);

        var fPlusH = interpolator.Interpolate(xPlusH);
        var fCenter = interpolator.Interpolate(x);
        var fMinusH = interpolator.Interpolate(xMinusH);

        // f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
        var numerator = numOps.Add(
            numOps.Subtract(fPlusH, numOps.Multiply(numOps.FromDouble(2), fCenter)),
            fMinusH);
        var denominator = numOps.FromDouble(h * h);

        return Convert.ToDouble(numOps.Divide(numerator, denominator));
    }

    /// <summary>
    /// Checks if a sequence is monotonically increasing.
    /// </summary>
    public static bool IsMonotonicallyIncreasing(double[] values, double tolerance = 1e-10)
    {
        for (int i = 0; i < values.Length - 1; i++)
        {
            if (values[i + 1] < values[i] - tolerance)
                return false;
        }
        return true;
    }

    /// <summary>
    /// Checks if a sequence is monotonically decreasing.
    /// </summary>
    public static bool IsMonotonicallyDecreasing(double[] values, double tolerance = 1e-10)
    {
        for (int i = 0; i < values.Length - 1; i++)
        {
            if (values[i + 1] > values[i] + tolerance)
                return false;
        }
        return true;
    }

    /// <summary>
    /// Creates sample data from a mathematical function.
    /// </summary>
    public static (Vector<double> x, Vector<double> y) CreateFunctionData(
        Func<double, double> function,
        double xMin,
        double xMax,
        int numPoints)
    {
        var x = new Vector<double>(numPoints);
        var y = new Vector<double>(numPoints);

        for (int i = 0; i < numPoints; i++)
        {
            x[i] = xMin + (xMax - xMin) * i / (numPoints - 1);
            y[i] = function(x[i]);
        }

        return (x, y);
    }
}
```

### Phase 2: Implement Tests for Each Spline Type (10-15 hours)

I'll provide detailed test templates for the most important ones.

#### AC 2.1: CubicSplineInterpolation Tests (2 hours)

**File**: `CubicSplineInterpolationTests.cs`

```csharp
namespace AiDotNetTests.UnitTests.Interpolation;

using AiDotNet.Interpolation;
using AiDotNet;
using Xunit;

public class CubicSplineInterpolationTests
{
    #region Constructor Tests

    [Fact]
    public void Constructor_WithValidData_CreatesInstance()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });

        var spline = new CubicSplineInterpolation<double>(x, y);

        Assert.NotNull(spline);
    }

    [Fact]
    public void Constructor_WithTooFewPoints_ThrowsArgumentException()
    {
        var x = new Vector<double>(new[] { 0.0, 1.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0 });

        // Cubic spline requires at least 3 points (verify actual requirement)
        var ex = Assert.Throws<ArgumentException>(() =>
            new CubicSplineInterpolation<double>(x, y));
    }

    #endregion

    #region Exact Point Tests

    [Theory]
    [InlineData(0.0, 0.0)]
    [InlineData(1.0, 1.0)]
    [InlineData(2.0, 0.0)]
    [InlineData(3.0, 1.0)]
    public void Interpolate_AtKnownPoint_ReturnsExactValue(double x, double expectedY)
    {
        var xData = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var yData = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var spline = new CubicSplineInterpolation<double>(xData, yData);

        double result = spline.Interpolate(x);

        Assert.Equal(expectedY, result, precision: 10);
    }

    #endregion

    #region Smoothness Tests

    [Fact]
    public void Interpolate_HasC1Continuity()
    {
        // Arrange
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        var spline = new CubicSplineInterpolation<double>(x, y);

        // Act & Assert - Test derivative continuity at each data point
        for (int i = 1; i < x.Length - 1; i++)
        {
            double leftDeriv = SplineTestHelpers.ComputeFirstDerivative(spline, x[i] - 1e-6);
            double rightDeriv = SplineTestHelpers.ComputeFirstDerivative(spline, x[i] + 1e-6);

            Assert.Equal(leftDeriv, rightDeriv, precision: 3);
        }
    }

    [Fact]
    public void Interpolate_HasC2Continuity()
    {
        // Arrange
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.5, 1.5, 1.0 });
        var spline = new CubicSplineInterpolation<double>(x, y);

        // Act & Assert - Test second derivative continuity at each interior point
        for (int i = 1; i < x.Length - 1; i++)
        {
            double leftSecondDeriv = SplineTestHelpers.ComputeSecondDerivative(spline, x[i] - 1e-5);
            double rightSecondDeriv = SplineTestHelpers.ComputeSecondDerivative(spline, x[i] + 1e-5);

            Assert.Equal(leftSecondDeriv, rightSecondDeriv, precision: 2);
        }
    }

    #endregion

    #region Natural Boundary Condition Tests

    [Fact]
    public void NaturalSpline_HasZeroSecondDerivativeAtBoundaries()
    {
        // Arrange
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 0.5, 1.5 });
        var spline = new CubicSplineInterpolation<double>(x, y);

        // Act - Compute second derivatives at boundaries
        double secondDerivAtStart = SplineTestHelpers.ComputeSecondDerivative(spline, x[0]);
        double secondDerivAtEnd = SplineTestHelpers.ComputeSecondDerivative(spline, x[x.Length - 1]);

        // Assert - Should be approximately zero (natural boundary condition)
        Assert.Equal(0.0, secondDerivAtStart, precision: 2);
        Assert.Equal(0.0, secondDerivAtEnd, precision: 2);
    }

    #endregion

    #region Known Function Tests

    [Fact]
    public void Interpolate_QuadraticFunction_ExactResults()
    {
        // Arrange - Cubic spline is exact for polynomials of degree <= 3
        // Test with y = x²
        var (x, y) = SplineTestHelpers.CreateFunctionData(
            t => t * t,
            xMin: 0.0,
            xMax: 5.0,
            numPoints: 10);

        var spline = new CubicSplineInterpolation<double>(x, y);

        // Act & Assert - Test at many intermediate points
        for (double testX = 0.0; testX <= 5.0; testX += 0.1)
        {
            double interpolated = spline.Interpolate(testX);
            double expected = testX * testX;

            Assert.Equal(expected, interpolated, precision: 8);
        }
    }

    [Fact]
    public void Interpolate_SineFunction_HighAccuracy()
    {
        // Arrange - Sample sine wave
        var (x, y) = SplineTestHelpers.CreateFunctionData(
            Math.Sin,
            xMin: 0.0,
            xMax: Math.PI,
            numPoints: 20);

        var spline = new CubicSplineInterpolation<double>(x, y);

        // Act & Assert - Test accuracy at intermediate points
        for (double testX = 0.1; testX < Math.PI; testX += 0.1)
        {
            double interpolated = spline.Interpolate(testX);
            double expected = Math.Sin(testX);

            Assert.Equal(expected, interpolated, precision: 3);
        }
    }

    #endregion
}
```

#### AC 2.2: MonotoneCubicInterpolation Tests (2 hours)

**File**: `MonotoneCubicInterpolationTests.cs`

**Critical Test**: Monotonicity Preservation

```csharp
[Fact]
public void Interpolate_WithIncreasingData_PreservesMonotonicity()
{
    // Arrange - Strictly increasing data
    var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
    var y = new Vector<double>(new[] { 10.0, 15.0, 18.0, 25.0, 30.0 });
    var spline = new MonotoneCubicInterpolation<double>(x, y);

    // Act - Sample many points
    var samples = new List<double>();
    for (double testX = 0.0; testX <= 4.0; testX += 0.05)
    {
        samples.Add(spline.Interpolate(testX));
    }

    // Assert - All samples should be monotonically increasing
    Assert.True(SplineTestHelpers.IsMonotonicallyIncreasing(samples.ToArray()),
        "Monotonicity not preserved");
}

[Fact]
public void Interpolate_WithDecreasingData_PreservesMonotonicity()
{
    // Arrange - Strictly decreasing data
    var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
    var y = new Vector<double>(new[] { 30.0, 25.0, 18.0, 15.0, 10.0 });
    var spline = new MonotoneCubicInterpolation<double>(x, y);

    // Act - Sample many points
    var samples = new List<double>();
    for (double testX = 0.0; testX <= 4.0; testX += 0.05)
    {
        samples.Add(spline.Interpolate(testX));
    }

    // Assert - All samples should be monotonically decreasing
    Assert.True(SplineTestHelpers.IsMonotonicallyDecreasing(samples.ToArray()),
        "Monotonicity not preserved");
}

[Fact]
public void Interpolate_WithMixedMonotoneRegions_PreservesLocalMonotonicity()
{
    // Arrange - Data with different monotone regions
    var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 });
    var y = new Vector<double>(new[] { 10.0, 20.0, 25.0, 20.0, 15.0, 10.0 });
    //                                  ^increasing^  ^decreasing^
    var spline = new MonotoneCubicInterpolation<double>(x, y);

    // Act & Assert - Check increasing region [0, 2]
    for (double testX = 0.0; testX < 2.0; testX += 0.1)
    {
        double y1 = spline.Interpolate(testX);
        double y2 = spline.Interpolate(testX + 0.1);
        Assert.True(y2 >= y1, $"Not monotone at x={testX}");
    }

    // Act & Assert - Check decreasing region [2, 5]
    for (double testX = 2.0; testX < 5.0; testX += 0.1)
    {
        double y1 = spline.Interpolate(testX);
        double y2 = spline.Interpolate(testX + 0.1);
        Assert.True(y2 <= y1, $"Not monotone at x={testX}");
    }
}
```

#### AC 2.3-2.9: Tests for Other Spline Types

Follow similar patterns for:
- **CubicBSplineInterpolation**: Test smoothness, local control
- **CatmullRomSplineInterpolation**: Test tangent calculation, passes through points
- **HermiteInterpolation**: Test with specified derivatives
- **AkimaInterpolation**: Test overshoot avoidance, minimum 5 points
- **ClampedSplineInterpolation**: Test specified boundary derivatives
- **AdaptiveCubicSplineInterpolation**: Test adaptation to data
- **KochanekBartelsSplineInterpolation**: Test T/C/B parameters

### Phase 3: Run and Debug Tests (2-3 hours)

#### AC 3.1: Run All Spline Tests

```bash
cd C:/Users/cheat/source/repos/AiDotNet
dotnet test tests/AiDotNetTests.csproj --filter "FullyQualifiedName~Spline"
```

#### AC 3.2: Measure Coverage

```bash
dotnet test tests/AiDotNetTests.csproj \
  --filter "FullyQualifiedName~Interpolation" \
  --collect:"XPlat Code Coverage"
```

**Target**: 80%+ coverage for each spline class.

### Phase 4: Documentation (30 minutes)

Add XML comments documenting test strategy and mathematical properties being verified.

---

## Mathematical Background

### Cubic Polynomial Form

Each spline segment is a cubic polynomial:
```
S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)² + d_i*(x-x_i)³
```

### Continuity Conditions

- **C0**: Function continuous (no jumps)
- **C1**: First derivative continuous (no kinks)
- **C2**: Second derivative continuous (no sudden curvature changes)

### Natural Spline Boundary Conditions

```
S''(x_0) = 0  (zero curvature at start)
S''(x_n) = 0  (zero curvature at end)
```

### Monotonicity

A function f is monotonically increasing if:
```
x1 < x2  implies  f(x1) <= f(x2)
```

For monotone cubic interpolation, this property is preserved in each monotone region of the original data.

---

## Common Pitfalls

### Pitfall 1: Not Enough Data Points

Different splines have different minimum requirements:
- Cubic Spline: 3 points
- Akima: 5 points
- B-Spline: 4 points

Test with minimum and minimum-1 points.

### Pitfall 2: Assuming B-Splines Interpolate

B-splines **approximate** (don't necessarily pass through points).

Test differently than interpolating splines.

### Pitfall 3: Numerical Derivative Computation

Use appropriate step size `h`:
- Too large: Inaccurate
- Too small: Roundoff errors

Use `h = 1e-6` for first derivative, `h = 1e-4` for second derivative.

### Pitfall 4: Monotonicity Test Resolution

Test with fine enough resolution to catch violations:
```csharp
// Wrong: Steps too large
for (double x = 0.0; x < 5.0; x += 1.0)  // Might miss non-monotonic regions

// Correct: Fine resolution
for (double x = 0.0; x < 5.0; x += 0.01)  // Catches violations
```

---

## Checklist Summary

### Phase 1: Setup (1 hour)
- [ ] Create test files for all 9 spline types
- [ ] Create SplineTestHelpers class
- [ ] Verify tests compile

### Phase 2: Implement Tests (10-15 hours)
- [ ] CubicSplineInterpolation tests
- [ ] MonotoneCubicInterpolation tests (with monotonicity verification)
- [ ] AkimaInterpolation tests (with overshoot tests)
- [ ] HermiteInterpolation tests
- [ ] CatmullRomSplineInterpolation tests
- [ ] CubicBSplineInterpolation tests
- [ ] ClampedSplineInterpolation tests
- [ ] AdaptiveCubicSplineInterpolation tests
- [ ] KochanekBartelsSplineInterpolation tests

### Phase 3: Validation (2-3 hours)
- [ ] All tests pass
- [ ] Coverage 80%+
- [ ] Smoothness verified
- [ ] Monotonicity verified (where applicable)

### Phase 4: Documentation (30 minutes)
- [ ] XML comments on test classes
- [ ] Mathematical properties documented

### Total Estimated Time: 13-19 hours

---

## Success Criteria

1. **All Tests Pass**: 100% of tests pass
2. **High Coverage**: 80%+ code coverage
3. **Smoothness Verified**: C1/C2 continuity tested
4. **Monotonicity Verified**: Monotone splines preserve monotonicity
5. **Boundary Conditions**: Natural/clamped boundaries verified
6. **Documentation**: Test strategy clearly documented

---

## Resources

- **Cubic Spline**: https://en.wikipedia.org/wiki/Spline_(mathematics)
- **Monotone Interpolation**: https://en.wikipedia.org/wiki/Monotone_cubic_interpolation
- **Akima Spline**: https://en.wikipedia.org/wiki/Akima_spline
- **Hermite Spline**: https://en.wikipedia.org/wiki/Cubic_Hermite_spline
- **Kochanek-Bartels**: https://en.wikipedia.org/wiki/Kochanek%E2%80%93Bartels_spline
