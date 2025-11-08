# Issue #352: Junior Developer Implementation Guide
## Basic Interpolation - Unit Tests and Validation

---

## Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [What is Interpolation?](#what-is-interpolation)
3. [Understanding Basic Interpolation Methods](#understanding-basic-interpolation-methods)
4. [Current Implementation Status](#current-implementation-status)
5. [Testing Strategy](#testing-strategy)
6. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
7. [Mathematical Background](#mathematical-background)
8. [Common Pitfalls](#common-pitfalls)

---

## Understanding the Problem

### What Are We Solving?

The AiDotNet library has **four basic interpolation implementations** that currently have **0% test coverage**. We need to create comprehensive unit tests that verify numerical accuracy, boundary conditions, and correctness.

### The Core Issue

**Currently**: The interpolation methods exist in the codebase but lack validation:
- No tests verify interpolation accuracy
- No tests check boundary conditions
- No tests validate error handling
- Developers have no confidence these methods work correctly

**Goal**: Create comprehensive test suites that:
1. Verify numerical accuracy against known functions
2. Test boundary conditions (edges of data range)
3. Test error handling (invalid inputs)
4. Provide confidence for production use

### Methods to Test (Issue #352)

1. **LinearInterpolation** - Straight lines between points
2. **BilinearInterpolation** - 2D grid interpolation
3. **NearestNeighborInterpolation** - Snap to closest known value
4. **Interpolation2DTo1DAdapter** - Convert 2D to 1D interpolation

---

## What is Interpolation?

### Real-World Analogy

Imagine you have temperature measurements:
- 9:00 AM: 65 degrees F
- 12:00 PM: 75 degrees F
- 3:00 PM: 70 degrees F

What was the temperature at 10:30 AM? You don't have a measurement, so you **interpolate** - you make an educated guess based on surrounding values.

### In AI and Data Processing

Interpolation is used for:
- **Image Resizing**: Creating new pixels between existing ones
- **Signal Processing**: Estimating values in time series data
- **Data Smoothing**: Filling gaps in sensor readings
- **Feature Engineering**: Generating intermediate values for ML models

---

## Understanding Basic Interpolation Methods

### 1. Linear Interpolation (1D)

**What it does**: Draws straight lines between known points.

**Formula**:
```
y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
```

**Visual Example**:
```
Known points:  (1, 10)  and  (5, 30)
Want to find:  y at x = 3

Calculation:
y = 10 + (3 - 1) * (30 - 10) / (5 - 1)
y = 10 + 2 * 20 / 4
y = 10 + 10
y = 20  <-- The interpolated value
```

**Properties**:
- Continuous (no jumps)
- Exact at known points
- Linear between points (straight lines)
- Simple and fast

**Use Cases**:
- Quick estimates
- Linear data trends
- Performance-critical applications

### 2. Bilinear Interpolation (2D)

**What it does**: Extends linear interpolation to 2D grids.

**How it works**:
1. Interpolate along x-direction twice (top and bottom)
2. Interpolate along y-direction once (using results from step 1)

**Visual Example** (2x2 grid):
```
Grid:
  (0,1)=12    (1,1)=16
      Q12----------Q22
       |            |
       |   (0.5,   |
       |   0.5)=?  |
       |            |
      Q11----------Q21
  (0,0)=10    (1,0)=14

Step 1: Interpolate along x (bottom edge, y=0):
  R1 = 10 + (0.5)*(14-10) = 12

Step 2: Interpolate along x (top edge, y=1):
  R2 = 12 + (0.5)*(16-12) = 14

Step 3: Interpolate along y (using R1 and R2):
  Result = 12 + (0.5)*(14-12) = 13
```

**Properties**:
- Continuous in both dimensions
- Exact at grid points
- Requires at least 2x2 grid
- Common in image processing

**Use Cases**:
- Image resizing (upscaling/downscaling)
- Texture mapping in graphics
- Geographic data interpolation

### 3. Nearest Neighbor Interpolation

**What it does**: Uses the value of the closest known point (no averaging).

**Algorithm**:
1. Find the distance to each known point
2. Pick the point with minimum distance
3. Return that point's value

**Visual Example**:
```
Known points:  (1, 10), (3, 20), (5, 15)
Want to find:  y at x = 2.3

Distances:
- Distance to (1, 10): |2.3 - 1| = 1.3
- Distance to (3, 20): |2.3 - 3| = 0.7  <-- Minimum!
- Distance to (5, 15): |2.3 - 5| = 2.7

Result: y = 20 (value at x=3)
```

**Properties**:
- Not continuous (can have jumps)
- Exact at known points
- Very fast (no arithmetic, just comparisons)
- Preserves original values (no blending)

**Use Cases**:
- Image resizing when preserving sharp edges
- Categorical data (can't average categories)
- Fast preview rendering

### 4. Interpolation2DTo1DAdapter

**What it does**: Allows using 2D interpolation for 1D data.

**How it works**:
- Takes a 2D interpolation object
- Fixes one dimension (e.g., y = 0)
- Exposes a 1D interface

**Use Case**:
- Code reuse: Use one 2D implementation for both 1D and 2D tasks
- Testing: Validate 2D interpolators using 1D test cases

---

## Current Implementation Status

### Existing Files

**Source Files** (all in `C:\Users\cheat\source\repos\AiDotNet\src\Interpolation\`):
1. `LinearInterpolation.cs` - Implements `IInterpolation<T>`
2. `BilinearInterpolation.cs` - Implements `I2DInterpolation<T>`
3. `NearestNeighborInterpolation.cs` - Implements `IInterpolation<T>`
4. `Interpolation2DTo1DAdapter.cs` - Wraps `I2DInterpolation<T>` as `IInterpolation<T>`

**Interface Files**:
- `C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IInterpolation.cs`
- `C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\I2DInterpolation.cs`

**Test Status**:
- **Current**: 0% coverage (no tests exist)
- **Target**: 80%+ coverage with numerical accuracy verification

---

## Testing Strategy

### Test Categories

#### 1. Exact Point Tests

**Purpose**: Verify interpolator returns exact values at known data points.

**Example**:
```csharp
[Fact]
public void Interpolate_AtKnownPoint_ReturnsExactValue()
{
    // Arrange
    var x = new Vector<double>(new[] { 1.0, 3.0, 5.0 });
    var y = new Vector<double>(new[] { 10.0, 20.0, 15.0 });
    var interpolator = new LinearInterpolation<double>(x, y);

    // Act
    double result = interpolator.Interpolate(3.0);

    // Assert
    Assert.Equal(20.0, result, precision: 5);
}
```

**What to test**:
- All known data points
- First and last points (boundary cases)
- Middle points

#### 2. Midpoint Tests

**Purpose**: Verify interpolation accuracy at midpoints (easy to calculate by hand).

**Example** (Linear):
```csharp
[Fact]
public void Interpolate_AtMidpoint_ReturnsAverageValue()
{
    // Arrange
    var x = new Vector<double>(new[] { 0.0, 10.0 });
    var y = new Vector<double>(new[] { 100.0, 200.0 });
    var interpolator = new LinearInterpolation<double>(x, y);

    // Act - midpoint x=5.0 should give y=150.0
    double result = interpolator.Interpolate(5.0);

    // Assert
    Assert.Equal(150.0, result, precision: 5);
}
```

#### 3. Known Function Tests

**Purpose**: Test against mathematical functions we can verify.

**Example** (Linear function y = 2x + 5):
```csharp
[Theory]
[InlineData(0.0, 5.0)]    // y = 2*0 + 5 = 5
[InlineData(1.0, 7.0)]    // y = 2*1 + 5 = 7
[InlineData(2.5, 10.0)]   // y = 2*2.5 + 5 = 10
[InlineData(5.0, 15.0)]   // y = 2*5 + 5 = 15
public void Interpolate_LinearFunction_ReturnsExactValues(double x, double expectedY)
{
    // Arrange - Create data points from y = 2x + 5
    var xData = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 });
    var yData = new Vector<double>(new[] { 5.0, 7.0, 9.0, 11.0, 13.0, 15.0 });
    var interpolator = new LinearInterpolation<double>(xData, yData);

    // Act
    double result = interpolator.Interpolate(x);

    // Assert
    Assert.Equal(expectedY, result, precision: 10);
}
```

#### 4. Boundary Condition Tests

**Purpose**: Test behavior at and beyond data boundaries.

**What to test**:
- At first data point
- Before first data point (extrapolation)
- At last data point
- Beyond last data point (extrapolation)

**Example**:
```csharp
[Theory]
[InlineData(-1.0)]  // Before first point
[InlineData(0.0)]   // At first point
[InlineData(10.0)]  // At last point
[InlineData(11.0)]  // Beyond last point
public void Interpolate_AtBoundaries_DoesNotThrow(double x)
{
    // Arrange
    var xData = new Vector<double>(new[] { 0.0, 5.0, 10.0 });
    var yData = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
    var interpolator = new LinearInterpolation<double>(xData, yData);

    // Act & Assert - Should not throw
    double result = interpolator.Interpolate(x);
    Assert.True(double.IsFinite(result));
}
```

#### 5. Input Validation Tests

**Purpose**: Verify error handling for invalid inputs.

**What to test**:
- Null vectors
- Empty vectors
- Mismatched vector lengths
- Unsorted x-coordinates
- Duplicate x-coordinates

**Example**:
```csharp
[Fact]
public void Constructor_WithMismatchedVectorLengths_ThrowsArgumentException()
{
    // Arrange
    var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
    var y = new Vector<double>(new[] { 10.0, 20.0 });  // Wrong length!

    // Act & Assert
    var ex = Assert.Throws<ArgumentException>(() =>
        new LinearInterpolation<double>(x, y));

    Assert.Contains("same length", ex.Message, StringComparison.OrdinalIgnoreCase);
}
```

#### 6. Numerical Precision Tests

**Purpose**: Verify results are numerically accurate.

**Approach**:
- Use double precision
- Compare with tolerance (e.g., 1e-10)
- Test extreme values (very large/small)

**Example**:
```csharp
[Fact]
public void Interpolate_WithSmallValues_MaintainsPrecision()
{
    // Arrange - Test with very small values
    var x = new Vector<double>(new[] { 1e-10, 2e-10, 3e-10 });
    var y = new Vector<double>(new[] { 1e-20, 2e-20, 3e-20 });
    var interpolator = new LinearInterpolation<double>(x, y);

    // Act
    double result = interpolator.Interpolate(1.5e-10);

    // Assert
    Assert.Equal(1.5e-20, result, precision: 25);
}
```

---

## Step-by-Step Implementation Guide

### Phase 1: Set Up Test Infrastructure (30 minutes)

#### AC 1.1: Create Test File for LinearInterpolation

**File**: `C:\Users\cheat\source\repos\AiDotNet\tests\UnitTests\Interpolation\LinearInterpolationTests.cs`

```csharp
namespace AiDotNetTests.UnitTests.Interpolation;

using AiDotNet.Interpolation;
using AiDotNet;
using Xunit;

/// <summary>
/// Unit tests for LinearInterpolation class.
/// Tests verify numerical accuracy, boundary conditions, and error handling.
/// </summary>
public class LinearInterpolationTests
{
    #region Constructor Tests

    [Fact]
    public void Constructor_WithValidData_CreatesInstance()
    {
        // Arrange
        var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 10.0, 20.0, 30.0 });

        // Act
        var interpolator = new LinearInterpolation<double>(x, y);

        // Assert
        Assert.NotNull(interpolator);
    }

    [Fact]
    public void Constructor_WithMismatchedLengths_ThrowsArgumentException()
    {
        // Arrange
        var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var y = new Vector<double>(new[] { 10.0, 20.0 });  // Length mismatch

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() =>
            new LinearInterpolation<double>(x, y));

        Assert.Contains("same length", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    #endregion

    #region Exact Point Tests

    [Theory]
    [InlineData(1.0, 10.0)]
    [InlineData(2.0, 20.0)]
    [InlineData(3.0, 30.0)]
    public void Interpolate_AtKnownPoint_ReturnsExactValue(double x, double expectedY)
    {
        // Arrange
        var xData = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var yData = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
        var interpolator = new LinearInterpolation<double>(xData, yData);

        // Act
        double result = interpolator.Interpolate(x);

        // Assert
        Assert.Equal(expectedY, result, precision: 10);
    }

    #endregion

    #region Midpoint Tests

    [Fact]
    public void Interpolate_AtMidpoint_ReturnsAverageValue()
    {
        // Arrange
        var x = new Vector<double>(new[] { 0.0, 10.0 });
        var y = new Vector<double>(new[] { 100.0, 200.0 });
        var interpolator = new LinearInterpolation<double>(x, y);

        // Act - Midpoint x=5.0 should give y=150.0
        double result = interpolator.Interpolate(5.0);

        // Assert
        Assert.Equal(150.0, result, precision: 10);
    }

    [Theory]
    [InlineData(1.5, 15.0)]  // Midpoint between (1,10) and (2,20)
    [InlineData(2.5, 25.0)]  // Midpoint between (2,20) and (3,30)
    public void Interpolate_AtMultipleMidpoints_ReturnsCorrectValues(double x, double expectedY)
    {
        // Arrange
        var xData = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var yData = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
        var interpolator = new LinearInterpolation<double>(xData, yData);

        // Act
        double result = interpolator.Interpolate(x);

        // Assert
        Assert.Equal(expectedY, result, precision: 10);
    }

    #endregion

    #region Known Function Tests

    [Theory]
    [InlineData(0.0, 5.0)]    // y = 2*0 + 5 = 5
    [InlineData(0.5, 6.0)]    // y = 2*0.5 + 5 = 6
    [InlineData(1.0, 7.0)]    // y = 2*1 + 5 = 7
    [InlineData(2.5, 10.0)]   // y = 2*2.5 + 5 = 10
    [InlineData(5.0, 15.0)]   // y = 2*5 + 5 = 15
    public void Interpolate_LinearFunction_ReturnsExactValues(double x, double expectedY)
    {
        // Arrange - Create data points from y = 2x + 5
        var xData = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 });
        var yData = new Vector<double>(new[] { 5.0, 7.0, 9.0, 11.0, 13.0, 15.0 });
        var interpolator = new LinearInterpolation<double>(xData, yData);

        // Act
        double result = interpolator.Interpolate(x);

        // Assert
        Assert.Equal(expectedY, result, precision: 10);
    }

    #endregion

    #region Boundary Condition Tests

    [Fact]
    public void Interpolate_AtFirstPoint_ReturnsFirstValue()
    {
        // Arrange
        var x = new Vector<double>(new[] { 0.0, 5.0, 10.0 });
        var y = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
        var interpolator = new LinearInterpolation<double>(x, y);

        // Act
        double result = interpolator.Interpolate(0.0);

        // Assert
        Assert.Equal(10.0, result, precision: 10);
    }

    [Fact]
    public void Interpolate_AtLastPoint_ReturnsLastValue()
    {
        // Arrange
        var x = new Vector<double>(new[] { 0.0, 5.0, 10.0 });
        var y = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
        var interpolator = new LinearInterpolation<double>(x, y);

        // Act
        double result = interpolator.Interpolate(10.0);

        // Assert
        Assert.Equal(30.0, result, precision: 10);
    }

    [Fact]
    public void Interpolate_BeforeFirstPoint_DoesNotThrow()
    {
        // Arrange
        var x = new Vector<double>(new[] { 5.0, 10.0, 15.0 });
        var y = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
        var interpolator = new LinearInterpolation<double>(x, y);

        // Act - Extrapolate before first point
        double result = interpolator.Interpolate(0.0);

        // Assert - Should return first value (clamping behavior)
        Assert.Equal(10.0, result, precision: 10);
    }

    [Fact]
    public void Interpolate_AfterLastPoint_DoesNotThrow()
    {
        // Arrange
        var x = new Vector<double>(new[] { 5.0, 10.0, 15.0 });
        var y = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
        var interpolator = new LinearInterpolation<double>(x, y);

        // Act - Extrapolate after last point
        double result = interpolator.Interpolate(20.0);

        // Assert - Should return last value (clamping behavior)
        Assert.Equal(30.0, result, precision: 10);
    }

    #endregion

    #region Numerical Precision Tests

    [Fact]
    public void Interpolate_WithLargeValues_MaintainsPrecision()
    {
        // Arrange - Test with large values
        var x = new Vector<double>(new[] { 1e10, 2e10, 3e10 });
        var y = new Vector<double>(new[] { 1e15, 2e15, 3e15 });
        var interpolator = new LinearInterpolation<double>(x, y);

        // Act
        double result = interpolator.Interpolate(1.5e10);

        // Assert
        Assert.Equal(1.5e15, result, precision: 5);
    }

    [Fact]
    public void Interpolate_WithSmallValues_MaintainsPrecision()
    {
        // Arrange - Test with very small values
        var x = new Vector<double>(new[] { 1e-10, 2e-10, 3e-10 });
        var y = new Vector<double>(new[] { 1e-5, 2e-5, 3e-5 });
        var interpolator = new LinearInterpolation<double>(x, y);

        // Act
        double result = interpolator.Interpolate(1.5e-10);

        // Assert
        Assert.Equal(1.5e-5, result, precision: 15);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Interpolate_TwoPointsOnly_WorksCorrectly()
    {
        // Arrange - Minimum valid dataset
        var x = new Vector<double>(new[] { 0.0, 1.0 });
        var y = new Vector<double>(new[] { 0.0, 10.0 });
        var interpolator = new LinearInterpolation<double>(x, y);

        // Act
        double result = interpolator.Interpolate(0.5);

        // Assert
        Assert.Equal(5.0, result, precision: 10);
    }

    [Fact]
    public void Interpolate_WithNegativeValues_WorksCorrectly()
    {
        // Arrange
        var x = new Vector<double>(new[] { -10.0, -5.0, 0.0 });
        var y = new Vector<double>(new[] { -100.0, -50.0, 0.0 });
        var interpolator = new LinearInterpolation<double>(x, y);

        // Act
        double result = interpolator.Interpolate(-7.5);

        // Assert
        Assert.Equal(-75.0, result, precision: 10);
    }

    #endregion
}
```

#### AC 1.2: Create Directory Structure

```bash
# Create test directory if it doesn't exist
mkdir -p C:/Users/cheat/source/repos/AiDotNet/tests/UnitTests/Interpolation
```

### Phase 2: Implement Tests for Each Method (4-6 hours)

#### AC 2.1: LinearInterpolation Tests (1.5 hours)

**Task**: Complete the test file created in Phase 1.

**Checklist**:
- [ ] Constructor validation tests
- [ ] Exact point tests
- [ ] Midpoint tests
- [ ] Known function tests (linear function)
- [ ] Boundary condition tests
- [ ] Numerical precision tests
- [ ] Edge case tests

**Run tests**:
```bash
cd C:/Users/cheat/source/repos/AiDotNet
dotnet test tests/AiDotNetTests.csproj --filter "FullyQualifiedName~LinearInterpolationTests"
```

#### AC 2.2: BilinearInterpolation Tests (2 hours)

**File**: `C:\Users\cheat\source\repos\AiDotNet\tests\UnitTests\Interpolation\BilinearInterpolationTests.cs`

**Test Template**:
```csharp
namespace AiDotNetTests.UnitTests.Interpolation;

using AiDotNet.Interpolation;
using AiDotNet;
using Xunit;

public class BilinearInterpolationTests
{
    #region Constructor Tests

    [Fact]
    public void Constructor_WithValidData_CreatesInstance()
    {
        // Arrange
        var x = new Vector<double>(new[] { 0.0, 1.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0 });
        var z = new Matrix<double>(2, 2);
        z[0, 0] = 10.0; z[0, 1] = 12.0;
        z[1, 0] = 14.0; z[1, 1] = 16.0;

        // Act
        var interpolator = new BilinearInterpolation<double>(x, y, z);

        // Assert
        Assert.NotNull(interpolator);
    }

    [Fact]
    public void Constructor_WithDimensionMismatch_ThrowsArgumentException()
    {
        // Arrange
        var x = new Vector<double>(new[] { 0.0, 1.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0, 2.0 });  // Wrong length
        var z = new Matrix<double>(2, 2);

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() =>
            new BilinearInterpolation<double>(x, y, z));

        Assert.Contains("mismatch", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Constructor_WithTooSmallGrid_ThrowsArgumentException()
    {
        // Arrange - Less than 2x2 grid
        var x = new Vector<double>(new[] { 0.0 });
        var y = new Vector<double>(new[] { 0.0 });
        var z = new Matrix<double>(1, 1);

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() =>
            new BilinearInterpolation<double>(x, y, z));

        Assert.Contains("2x2", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    #endregion

    #region Corner Point Tests

    [Theory]
    [InlineData(0.0, 0.0, 10.0)]  // Bottom-left
    [InlineData(1.0, 0.0, 14.0)]  // Bottom-right
    [InlineData(0.0, 1.0, 12.0)]  // Top-left
    [InlineData(1.0, 1.0, 16.0)]  // Top-right
    public void Interpolate_AtGridPoint_ReturnsExactValue(double x, double y, double expectedZ)
    {
        // Arrange
        var xData = new Vector<double>(new[] { 0.0, 1.0 });
        var yData = new Vector<double>(new[] { 0.0, 1.0 });
        var zData = new Matrix<double>(2, 2);
        zData[0, 0] = 10.0; zData[0, 1] = 12.0;
        zData[1, 0] = 14.0; zData[1, 1] = 16.0;

        var interpolator = new BilinearInterpolation<double>(xData, yData, zData);

        // Act
        double result = interpolator.Interpolate(x, y);

        // Assert
        Assert.Equal(expectedZ, result, precision: 10);
    }

    #endregion

    #region Center Point Tests

    [Fact]
    public void Interpolate_AtCenterOfGrid_ReturnsAverageValue()
    {
        // Arrange - Simple 2x2 grid
        var x = new Vector<double>(new[] { 0.0, 1.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0 });
        var z = new Matrix<double>(2, 2);
        z[0, 0] = 10.0; z[0, 1] = 12.0;
        z[1, 0] = 14.0; z[1, 1] = 16.0;

        var interpolator = new BilinearInterpolation<double>(x, y, z);

        // Act - Center point should be average of all four corners
        double result = interpolator.Interpolate(0.5, 0.5);
        double expected = (10.0 + 12.0 + 14.0 + 16.0) / 4.0;

        // Assert
        Assert.Equal(expected, result, precision: 10);
    }

    #endregion

    #region Known Function Tests

    [Theory]
    [InlineData(0.0, 0.0, 5.0)]     // z = 2*0 + 3*0 + 5 = 5
    [InlineData(1.0, 0.0, 7.0)]     // z = 2*1 + 3*0 + 5 = 7
    [InlineData(0.0, 1.0, 8.0)]     // z = 2*0 + 3*1 + 5 = 8
    [InlineData(1.0, 1.0, 10.0)]    // z = 2*1 + 3*1 + 5 = 10
    [InlineData(0.5, 0.5, 7.75)]    // z = 2*0.5 + 3*0.5 + 5 = 7.5
    public void Interpolate_PlanarFunction_ReturnsExactValues(double x, double y, double expectedZ)
    {
        // Arrange - Create grid from planar function z = 2x + 3y + 5
        var xData = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        var yData = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        var zData = new Matrix<double>(3, 3);

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                zData[i, j] = 2.0 * xData[i] + 3.0 * yData[j] + 5.0;
            }
        }

        var interpolator = new BilinearInterpolation<double>(xData, yData, zData);

        // Act
        double result = interpolator.Interpolate(x, y);

        // Assert - Bilinear should be exact for planar functions
        Assert.Equal(expectedZ, result, precision: 10);
    }

    #endregion

    #region Boundary Condition Tests

    [Fact]
    public void Interpolate_AtGridBoundaries_ReturnsEdgeValues()
    {
        // Arrange
        var x = new Vector<double>(new[] { 0.0, 5.0, 10.0 });
        var y = new Vector<double>(new[] { 0.0, 5.0, 10.0 });
        var z = new Matrix<double>(3, 3);

        // Fill with values
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                z[i, j] = i * 10 + j;

        var interpolator = new BilinearInterpolation<double>(x, y, z);

        // Act & Assert - Test all four edges
        Assert.Equal(0.0, interpolator.Interpolate(0.0, 0.0), precision: 10);  // Bottom-left
        Assert.Equal(20.0, interpolator.Interpolate(10.0, 0.0), precision: 10); // Bottom-right
        Assert.Equal(2.0, interpolator.Interpolate(0.0, 10.0), precision: 10);  // Top-left
        Assert.Equal(22.0, interpolator.Interpolate(10.0, 10.0), precision: 10); // Top-right
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Interpolate_Minimum2x2Grid_WorksCorrectly()
    {
        // Arrange - Minimum valid grid size
        var x = new Vector<double>(new[] { 0.0, 1.0 });
        var y = new Vector<double>(new[] { 0.0, 1.0 });
        var z = new Matrix<double>(2, 2);
        z[0, 0] = 1.0; z[0, 1] = 2.0;
        z[1, 0] = 3.0; z[1, 1] = 4.0;

        var interpolator = new BilinearInterpolation<double>(x, y, z);

        // Act
        double result = interpolator.Interpolate(0.5, 0.5);

        // Assert - Should be average of corners
        Assert.Equal(2.5, result, precision: 10);
    }

    #endregion
}
```

**Checklist for BilinearInterpolation**:
- [ ] Constructor validation tests (dimension mismatch, too small grid)
- [ ] Corner point tests (all 4 corners)
- [ ] Center point tests
- [ ] Known function tests (planar function z = ax + by + c)
- [ ] Boundary condition tests
- [ ] Edge case tests (minimum 2x2 grid)

#### AC 2.3: NearestNeighborInterpolation Tests (1 hour)

**File**: `C:\Users\cheat\source\repos\AiDotNet\tests\UnitTests\Interpolation\NearestNeighborInterpolationTests.cs`

**Key Tests**:
```csharp
[Theory]
[InlineData(1.0, 10.0)]   // Exact match
[InlineData(1.3, 10.0)]   // Closer to 1.0 than 2.0
[InlineData(1.6, 20.0)]   // Closer to 2.0 than 1.0
[InlineData(2.4, 20.0)]   // Closer to 2.0 than 3.0
[InlineData(2.7, 30.0)]   // Closer to 3.0 than 2.0
public void Interpolate_ReturnsNearestValue(double x, double expectedY)
{
    // Arrange
    var xData = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
    var yData = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
    var interpolator = new NearestNeighborInterpolation<double>(xData, yData);

    // Act
    double result = interpolator.Interpolate(x);

    // Assert
    Assert.Equal(expectedY, result, precision: 10);
}
```

**Checklist**:
- [ ] Exact point tests
- [ ] Distance calculation tests
- [ ] Boundary behavior tests
- [ ] Tie-breaking tests (exactly between two points)
- [ ] Constructor validation tests

#### AC 2.4: Interpolation2DTo1DAdapter Tests (1 hour)

**File**: `C:\Users\cheat\source\repos\AiDotNet\tests\UnitTests\Interpolation\Interpolation2DTo1DAdapterTests.cs`

**Key Concept**: This adapter wraps a 2D interpolator to provide 1D interface.

**Test Template**:
```csharp
[Fact]
public void Interpolate_UsesUnderlyingInterpolator_Correctly()
{
    // Arrange - Create a simple 2D interpolator
    var x = new Vector<double>(new[] { 0.0, 1.0 });
    var y = new Vector<double>(new[] { 0.0, 1.0 });
    var z = new Matrix<double>(2, 2);
    z[0, 0] = 10.0; z[0, 1] = 12.0;
    z[1, 0] = 14.0; z[1, 1] = 16.0;

    var bilinear = new BilinearInterpolation<double>(x, y, z);
    var adapter = new Interpolation2DTo1DAdapter<double>(bilinear, fixedY: 0.5);

    // Act - Interpolate along x-axis at fixed y=0.5
    double result = adapter.Interpolate(0.5);

    // Assert - Should match 2D interpolation at (0.5, 0.5)
    double expected = bilinear.Interpolate(0.5, 0.5);
    Assert.Equal(expected, result, precision: 10);
}
```

**Checklist**:
- [ ] Adapter correctly delegates to 2D interpolator
- [ ] Fixed y-coordinate is used correctly
- [ ] All x-values produce correct results
- [ ] Constructor validation tests

### Phase 3: Run and Debug Tests (1-2 hours)

#### AC 3.1: Run All Interpolation Tests

```bash
cd C:/Users/cheat/source/repos/AiDotNet
dotnet test tests/AiDotNetTests.csproj --filter "FullyQualifiedName~Interpolation"
```

#### AC 3.2: Fix Any Failing Tests

**Common Issues**:

1. **Precision Errors**: Adjust tolerance in `Assert.Equal(expected, actual, precision: X)`
2. **Boundary Behavior**: Verify extrapolation behavior (clamping vs. linear extrapolation)
3. **Data Ordering**: Ensure x-coordinates are sorted ascending

#### AC 3.3: Measure Code Coverage

```bash
dotnet test tests/AiDotNetTests.csproj \
  --filter "FullyQualifiedName~Interpolation" \
  --collect:"XPlat Code Coverage"
```

**Target**: 80%+ line coverage for each interpolation class.

### Phase 4: Documentation and Review (30 minutes)

#### AC 4.1: Document Test Approach

Add XML comments to test classes explaining what is being tested and why.

**Example**:
```csharp
/// <summary>
/// Unit tests for LinearInterpolation class.
///
/// Test Strategy:
/// 1. Constructor validation - ensures proper error handling
/// 2. Exact point tests - verifies interpolation returns exact values at known points
/// 3. Midpoint tests - validates linear interpolation formula
/// 4. Known function tests - tests against y = 2x + 5 (linear function)
/// 5. Boundary tests - verifies behavior at data boundaries
/// 6. Precision tests - ensures numerical accuracy with large/small values
///
/// Expected Coverage: 90%+ of LinearInterpolation.cs
/// </summary>
public class LinearInterpolationTests
{
    // Tests...
}
```

#### AC 4.2: Create Test Summary

Document what was tested in a comment at the top of each test file.

---

## Mathematical Background

### Linear Interpolation Formula

Given two points `(x0, y0)` and `(x1, y1)`, find `y` at `x`:

```
t = (x - x0) / (x1 - x0)      // Normalized position (0 to 1)
y = y0 + t * (y1 - y0)        // Linear blend
  = (1 - t) * y0 + t * y1     // Alternative form
```

### Bilinear Interpolation Formula

Given four corners of a grid cell:
- `Q11 = (x1, y1, z11)` - Bottom-left
- `Q21 = (x2, y1, z21)` - Bottom-right
- `Q12 = (x1, y2, z12)` - Top-left
- `Q22 = (x2, y2, z22)` - Top-right

Find `z` at `(x, y)`:

```
Step 1: Interpolate along x-axis (bottom edge)
R1 = ((x2 - x) / (x2 - x1)) * z11 + ((x - x1) / (x2 - x1)) * z21

Step 2: Interpolate along x-axis (top edge)
R2 = ((x2 - x) / (x2 - x1)) * z12 + ((x - x1) / (x2 - x1)) * z22

Step 3: Interpolate along y-axis (using R1 and R2)
z = ((y2 - y) / (y2 - y1)) * R1 + ((y - y1) / (y2 - y1)) * R2
```

### Nearest Neighbor Algorithm

```
For each data point i:
  distance[i] = |x - x[i]|

minIndex = index where distance[minIndex] is minimum
result = y[minIndex]
```

---

## Common Pitfalls

### Pitfall 1: Floating-Point Comparison

**Problem**: Direct equality `==` fails due to rounding errors.

**Wrong**:
```csharp
Assert.Equal(expected, actual);  // May fail on 0.30000000000000004 vs 0.3
```

**Correct**:
```csharp
Assert.Equal(expected, actual, precision: 10);  // Tolerates small differences
```

### Pitfall 2: Unsorted Input Data

**Problem**: Interpolation assumes x-coordinates are sorted.

**Test**:
```csharp
[Fact]
public void Constructor_WithUnsortedData_ThrowsOrSorts()
{
    var x = new Vector<double>(new[] { 3.0, 1.0, 2.0 });  // Unsorted!
    var y = new Vector<double>(new[] { 30.0, 10.0, 20.0 });

    // Depending on implementation:
    // Option A: Throw ArgumentException
    // Option B: Sort internally

    // Test current behavior
}
```

### Pitfall 3: Extrapolation Assumptions

**Problem**: Not clear how interpolator behaves outside data range.

**Behaviors**:
- **Clamping**: Return first/last value
- **Linear extrapolation**: Continue the line beyond boundaries
- **Throw exception**: Reject out-of-range queries

**Test All Three**:
```csharp
[Fact]
public void Interpolate_BeforeFirstPoint_ReturnsFirstValue()
{
    // Test actual behavior and document it
}
```

### Pitfall 4: Grid Dimension Validation

**Problem**: Bilinear needs at least 2x2, bicubic needs 4x4.

**Test**:
```csharp
[Theory]
[InlineData(1, 1)]  // Too small
[InlineData(1, 2)]  // Too small
[InlineData(2, 1)]  // Too small
public void Constructor_WithInvalidGridSize_ThrowsArgumentException(int rows, int cols)
{
    // Test dimension validation
}
```

### Pitfall 5: Matrix Row/Column Order

**Problem**: Confusion about whether `z[i, j]` corresponds to `x[i], y[j]` or vice versa.

**Verify**:
```csharp
[Fact]
public void Interpolate_VerifiesRowColumnOrder()
{
    // Create asymmetric grid to test order
    var x = new Vector<double>(new[] { 0.0, 1.0 });
    var y = new Vector<double>(new[] { 0.0, 2.0 });
    var z = new Matrix<double>(2, 2);
    z[0, 0] = 100;  // x=0, y=0
    z[1, 0] = 200;  // x=1, y=0
    z[0, 1] = 300;  // x=0, y=2
    z[1, 1] = 400;  // x=1, y=2

    var interp = new BilinearInterpolation<double>(x, y, z);

    // Verify (0, 0) gives 100, (1, 0) gives 200, etc.
    Assert.Equal(100.0, interp.Interpolate(0.0, 0.0), precision: 10);
    Assert.Equal(200.0, interp.Interpolate(1.0, 0.0), precision: 10);
}
```

---

## Checklist Summary

### Phase 1: Setup (30 minutes)
- [ ] Create test directory `tests/UnitTests/Interpolation/`
- [ ] Create `LinearInterpolationTests.cs` with basic structure
- [ ] Verify tests compile and run

### Phase 2: Implement Tests (4-6 hours)
- [ ] LinearInterpolation tests complete
- [ ] BilinearInterpolation tests complete
- [ ] NearestNeighborInterpolation tests complete
- [ ] Interpolation2DTo1DAdapter tests complete

### Phase 3: Validation (1-2 hours)
- [ ] All tests pass
- [ ] Code coverage 80%+
- [ ] No precision errors
- [ ] Boundary conditions documented

### Phase 4: Documentation (30 minutes)
- [ ] XML comments on test classes
- [ ] Test strategy documented
- [ ] Mathematical formulas verified

### Total Estimated Time: 6-9 hours

---

## Success Criteria

1. **All Tests Pass**: 100% of implemented tests pass
2. **High Coverage**: 80%+ code coverage on interpolation classes
3. **Numerical Accuracy**: Tests verify results to at least 10 decimal places
4. **Boundary Testing**: All edge cases tested (first/last point, extrapolation)
5. **Error Handling**: Invalid inputs throw appropriate exceptions
6. **Documentation**: Test files have clear XML comments explaining approach

---

## Resources

- **AiDotNet Source Code**: `C:\Users\cheat\source\repos\AiDotNet\src\Interpolation\`
- **xUnit Documentation**: https://xunit.net/
- **Linear Interpolation**: https://en.wikipedia.org/wiki/Linear_interpolation
- **Bilinear Interpolation**: https://en.wikipedia.org/wiki/Bilinear_interpolation
- **Nearest Neighbor**: https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation
