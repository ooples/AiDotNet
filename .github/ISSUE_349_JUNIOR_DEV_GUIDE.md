# Issue #349: Junior Developer Implementation Guide
## Unit Tests for Math/Statistics Helpers

---

## Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [Helper Classes Overview](#helper-classes-overview)
3. [Testing Strategy](#testing-strategy)
4. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
5. [Complete Test Examples](#complete-test-examples)

---

## Understanding the Problem

### What Are We Testing?

The `src/Helpers/` directory contains **mathematical and statistical utility classes** with **0% test coverage**. These helpers provide core mathematical operations used throughout the AI library.

### Why Is This Important?

These helpers are fundamental building blocks. If they have bugs:
- **Models produce wrong results** - incorrect math = incorrect predictions
- **Training fails** - numerical instability in algorithms
- **Statistics are misleading** - wrong metrics hide problems
- **Matrix operations crash** - singular matrices, dimension mismatches

### Files to Test

```
src/Helpers/
├── MathHelper.cs              # Core math functions (987 lines)
├── StatisticsHelper.cs        # Statistical calculations
├── MatrixHelper.cs            # Matrix operations
└── MatrixSolutionHelper.cs    # Matrix equation solving
```

---

## Helper Classes Overview

### MathHelper

**Purpose**: Provides mathematical utility functions for AI algorithms

**Key Functions**:
1. **Numeric Operations**
   - `GetNumericOperations<T>()` - Get operations for type T
   - `Clamp<T>(value, min, max)` - Constrain value to range

2. **Special Functions**
   - `BesselI0<T>(x)` - Modified Bessel function (for von Mises distribution)
   - `BesselJ<T>(nu, x)` - Bessel function (for signal processing)
   - `BesselK<T>(nu, x)` - Modified Bessel function (for probability)
   - `Gamma<T>(x)` - Gamma function (for statistics)
   - `Erf<T>(x)` - Error function (for normal distribution)

3. **Common Math**
   - `Sigmoid<T>(x)` - Activation function: 1/(1 + e^-x)
   - `Tanh<T>(x)` - Hyperbolic tangent activation
   - `Sinc<T>(x)` - Sinc function: sin(πx)/(πx)
   - `Factorial<T>(n)` - Factorial function

4. **Trigonometry**
   - `Sin<T>(x)`, `Cos<T>(x)` - Trigonometric functions
   - `ArcSin<T>(x)`, `ArcCos<T>(x)`, `ArcTan<T>(x)` - Inverse trig

5. **Utilities**
   - `AlmostEqual<T>(a, b, tolerance)` - Floating-point equality
   - `GetNormalRandom<T>(mean, stdDev)` - Generate normal random values
   - `Log2(x)` - Base-2 logarithm
   - `Min<T>(a, b)`, `Max<T>(a, b)` - Comparison
   - `Modulo<T>(x, y)` - Modulo operation
   - `Reciprocal<T>(value)` - Calculate 1/x
   - `IsInteger<T>(value)` - Check if value is whole number

**Critical to Test**:
- Type safety for different numeric types (double, float, decimal, int, etc.)
- Edge cases (division by zero, negative logs, domain errors)
- Numerical accuracy (tolerance for floating-point)
- Special function correctness (compared to known values)

### StatisticsHelper

**Purpose**: Statistical calculations for data analysis and model evaluation

**Key Functions** (assuming similar to typical stats helpers):
1. **Descriptive Statistics**
   - Mean, Median, Mode
   - Variance, Standard Deviation
   - Min, Max, Range

2. **Correlations**
   - Pearson correlation
   - Spearman correlation
   - Covariance

3. **Distributions**
   - Normal distribution PDF/CDF
   - Z-scores
   - Percentiles

**Critical to Test**:
- Accuracy of statistical calculations
- Handling of edge cases (empty data, single value, outliers)
- Numerical stability

### MatrixHelper

**Purpose**: Matrix operations for linear algebra

**Key Functions** (typical matrix operations):
1. **Basic Operations**
   - Matrix multiplication
   - Matrix transpose
   - Matrix addition/subtraction

2. **Decompositions**
   - LU decomposition
   - QR decomposition
   - Eigenvalue decomposition

3. **Properties**
   - Matrix rank
   - Determinant
   - Trace
   - Norm

**Critical to Test**:
- Dimension compatibility
- Numerical stability
- Singular matrix handling
- Performance with large matrices

### MatrixSolutionHelper

**Purpose**: Solving systems of linear equations

**Key Functions**:
1. **Linear Systems**
   - Solve Ax = b
   - Least squares solutions
   - Overdetermined/underdetermined systems

2. **Matrix Inversion**
   - Pseudoinverse (Moore-Penrose)
   - Regularized inverse

3. **Optimization**
   - QR solver
   - SVD solver
   - Iterative solvers

**Critical to Test**:
- Solution correctness (verify Ax = b)
- Singular matrix handling
- Numerical stability
- Different matrix types (square, rectangular, rank-deficient)

---

## Testing Strategy

### Coverage Goals

**MathHelper**:
- [ ] Numeric operations for all supported types (double, float, decimal, int, etc.)
- [ ] Special functions with known values (Bessel, Gamma, Erf)
- [ ] Edge cases (zero, negative, infinity, NaN)
- [ ] Trigonometric functions (all quadrants)
- [ ] Floating-point equality (tolerance testing)

**StatisticsHelper**:
- [ ] Descriptive statistics accuracy
- [ ] Edge cases (empty, single value, identical values)
- [ ] Correlation calculations
- [ ] Distribution functions

**MatrixHelper**:
- [ ] Matrix operations correctness
- [ ] Dimension validation
- [ ] Decompositions accuracy
- [ ] Singular matrix handling

**MatrixSolutionHelper**:
- [ ] Linear system solutions
- [ ] Overdetermined/underdetermined systems
- [ ] Pseudoinverse correctness
- [ ] Numerical stability

### Test File Structure

```
tests/Helpers/
├── MathHelperTests.cs
├── StatisticsHelperTests.cs
├── MatrixHelperTests.cs
└── MatrixSolutionHelperTests.cs
```

---

## Step-by-Step Implementation Guide

### Step 1: Create MathHelperTests.cs

**Test Categories**:

**1. Numeric Operations**
```csharp
[TestMethod]
public void GetNumericOperations_ForDouble_ReturnsDoubleOperations()
{
    // Act
    var ops = MathHelper.GetNumericOperations<double>();

    // Assert
    Assert.IsNotNull(ops);
    Assert.AreEqual(1.0, ops.One);
    Assert.AreEqual(0.0, ops.Zero);
}

[TestMethod]
[ExpectedException(typeof(NotSupportedException))]
public void GetNumericOperations_ForUnsupportedType_ThrowsException()
{
    // Act
    var ops = MathHelper.GetNumericOperations<string>();

    // Assert - Exception expected
}
```

**2. Clamp Function**
```csharp
[TestMethod]
public void Clamp_WithValueBelowMin_ReturnsMin()
{
    // Arrange
    double value = 5.0;
    double min = 10.0;
    double max = 20.0;

    // Act
    double result = MathHelper.Clamp(value, min, max);

    // Assert
    Assert.AreEqual(min, result);
}

[TestMethod]
public void Clamp_WithValueAboveMax_ReturnsMax()
{
    // Arrange
    double value = 25.0;
    double min = 10.0;
    double max = 20.0;

    // Act
    double result = MathHelper.Clamp(value, min, max);

    // Assert
    Assert.AreEqual(max, result);
}

[TestMethod]
public void Clamp_WithValueInRange_ReturnsValue()
{
    // Arrange
    double value = 15.0;
    double min = 10.0;
    double max = 20.0;

    // Act
    double result = MathHelper.Clamp(value, min, max);

    // Assert
    Assert.AreEqual(value, result);
}
```

**3. Sigmoid Function**
```csharp
[TestMethod]
public void Sigmoid_AtZero_ReturnsHalf()
{
    // Act
    double result = MathHelper.Sigmoid(0.0);

    // Assert
    Assert.AreEqual(0.5, result, 1e-10);
}

[TestMethod]
public void Sigmoid_WithLargePositive_ReturnsNearOne()
{
    // Act
    double result = MathHelper.Sigmoid(10.0);

    // Assert
    Assert.IsTrue(result > 0.999);
}

[TestMethod]
public void Sigmoid_WithLargeNegative_ReturnsNearZero()
{
    // Act
    double result = MathHelper.Sigmoid(-10.0);

    // Assert
    Assert.IsTrue(result < 0.001);
}
```

**4. Special Functions (Bessel I0)**
```csharp
[TestMethod]
public void BesselI0_AtZero_ReturnsOne()
{
    // Act
    double result = MathHelper.BesselI0(0.0);

    // Assert
    Assert.AreEqual(1.0, result, 1e-10);
}

[TestMethod]
public void BesselI0_WithKnownValue_ReturnsExpectedResult()
{
    // Arrange - I0(1) ≈ 1.266065877752
    double x = 1.0;
    double expected = 1.266065877752;

    // Act
    double result = MathHelper.BesselI0(x);

    // Assert
    Assert.AreEqual(expected, result, 1e-6);
}
```

**5. Gamma Function**
```csharp
[TestMethod]
public void Gamma_WithPositiveInteger_ReturnsFactorial()
{
    // Arrange - Gamma(5) = 4! = 24
    double x = 5.0;
    double expected = 24.0;

    // Act
    double result = MathHelper.Gamma(x);

    // Assert
    Assert.AreEqual(expected, result, 1e-6);
}

[TestMethod]
public void Gamma_AtHalf_ReturnsSquareRootOfPi()
{
    // Arrange - Gamma(0.5) = sqrt(π)
    double x = 0.5;
    double expected = Math.Sqrt(Math.PI);

    // Act
    double result = MathHelper.Gamma(x);

    // Assert
    Assert.AreEqual(expected, result, 1e-6);
}
```

**6. Trigonometric Functions**
```csharp
[TestMethod]
public void Sin_AtZero_ReturnsZero()
{
    // Act
    double result = MathHelper.Sin(0.0);

    // Assert
    Assert.AreEqual(0.0, result, 1e-10);
}

[TestMethod]
public void Sin_AtPiOverTwo_ReturnsOne()
{
    // Act
    double result = MathHelper.Sin(Math.PI / 2.0);

    // Assert
    Assert.AreEqual(1.0, result, 1e-10);
}

[TestMethod]
public void ArcSin_AtZero_ReturnsZero()
{
    // Act
    double result = MathHelper.ArcSin(0.0);

    // Assert
    Assert.AreEqual(0.0, result, 1e-10);
}

[TestMethod]
[ExpectedException(typeof(ArgumentOutOfRangeException))]
public void ArcSin_OutOfRange_ThrowsException()
{
    // Act
    double result = MathHelper.ArcSin(2.0);  // Out of [-1, 1] range

    // Assert - Exception expected
}
```

**7. Floating-Point Equality**
```csharp
[TestMethod]
public void AlmostEqual_WithIdenticalValues_ReturnsTrue()
{
    // Arrange
    double a = 1.0;
    double b = 1.0;

    // Act
    bool result = MathHelper.AlmostEqual(a, b);

    // Assert
    Assert.IsTrue(result);
}

[TestMethod]
public void AlmostEqual_WithinTolerance_ReturnsTrue()
{
    // Arrange
    double a = 1.0;
    double b = 1.0 + 1e-9;  // Within default tolerance 1e-8

    // Act
    bool result = MathHelper.AlmostEqual(a, b);

    // Assert
    Assert.IsTrue(result);
}

[TestMethod]
public void AlmostEqual_OutsideTolerance_ReturnsFalse()
{
    // Arrange
    double a = 1.0;
    double b = 1.1;
    double tolerance = 0.01;

    // Act
    bool result = MathHelper.AlmostEqual(a, b, tolerance);

    // Assert
    Assert.IsFalse(result);
}
```

**8. Factorial**
```csharp
[TestMethod]
public void Factorial_OfZero_ReturnsOne()
{
    // Act
    double result = MathHelper.Factorial<double>(0);

    // Assert
    Assert.AreEqual(1.0, result);
}

[TestMethod]
public void Factorial_OfFive_Returns120()
{
    // Act
    double result = MathHelper.Factorial<double>(5);

    // Assert
    Assert.AreEqual(120.0, result);
}
```

**9. Error Function**
```csharp
[TestMethod]
public void Erf_AtZero_ReturnsZero()
{
    // Act
    double result = MathHelper.Erf(0.0);

    // Assert
    Assert.AreEqual(0.0, result, 1e-10);
}

[TestMethod]
public void Erf_AtOne_ReturnsExpectedValue()
{
    // Arrange - erf(1) ≈ 0.8427
    double expected = 0.8427;

    // Act
    double result = MathHelper.Erf(1.0);

    // Assert
    Assert.AreEqual(expected, result, 1e-4);
}
```

**10. Normal Random**
```csharp
[TestMethod]
public void GetNormalRandom_WithMeanZero_GeneratesValuesAroundZero()
{
    // Arrange
    var values = new List<double>();

    // Act - Generate many values
    for (int i = 0; i < 1000; i++)
    {
        values.Add(MathHelper.GetNormalRandom<double>(0.0, 1.0));
    }

    // Assert - Mean should be close to 0
    double mean = values.Average();
    Assert.IsTrue(Math.Abs(mean) < 0.1,
        $"Mean should be close to 0, got {mean}");
}

[TestMethod]
public void GetNormalRandom_WithStdDevOne_HasCorrectVariance()
{
    // Arrange
    var values = new List<double>();

    // Act
    for (int i = 0; i < 1000; i++)
    {
        values.Add(MathHelper.GetNormalRandom<double>(0.0, 1.0));
    }

    // Assert
    double variance = values.Select(v => v * v).Average();
    Assert.IsTrue(variance > 0.8 && variance < 1.2,
        $"Variance should be ~1.0, got {variance}");
}
```

### Step 2: Create StatisticsHelperTests.cs

```csharp
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AiDotNet.Helpers;

namespace AiDotNetTests.Helpers;

[TestClass]
public class StatisticsHelperTests
{
    [TestMethod]
    public void Mean_WithValues_ReturnsCorrectMean()
    {
        // Arrange
        var values = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act
        double mean = StatisticsHelper.Mean(values);

        // Assert
        Assert.AreEqual(3.0, mean);
    }

    [TestMethod]
    public void StandardDeviation_WithKnownValues_ReturnsCorrectStdDev()
    {
        // Arrange
        var values = new[] { 2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0 };
        double expectedStdDev = 2.0;  // Known standard deviation

        // Act
        double stdDev = StatisticsHelper.StandardDeviation(values);

        // Assert
        Assert.AreEqual(expectedStdDev, stdDev, 0.1);
    }

    [TestMethod]
    public void Variance_WithKnownValues_ReturnsCorrectVariance()
    {
        // Arrange
        var values = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        double expectedVariance = 2.0;  // Variance of 1..5

        // Act
        double variance = StatisticsHelper.Variance(values);

        // Assert
        Assert.AreEqual(expectedVariance, variance, 0.1);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void Mean_WithEmptyArray_ThrowsException()
    {
        // Arrange
        var values = new double[0];

        // Act
        double mean = StatisticsHelper.Mean(values);

        // Assert - Exception expected
    }

    [TestMethod]
    public void Median_WithOddCount_ReturnsMiddleValue()
    {
        // Arrange
        var values = new[] { 1.0, 3.0, 2.0, 5.0, 4.0 };  // Sorted: 1,2,3,4,5

        // Act
        double median = StatisticsHelper.Median(values);

        // Assert
        Assert.AreEqual(3.0, median);
    }

    [TestMethod]
    public void Median_WithEvenCount_ReturnsAverage()
    {
        // Arrange
        var values = new[] { 1.0, 2.0, 3.0, 4.0 };

        // Act
        double median = StatisticsHelper.Median(values);

        // Assert
        Assert.AreEqual(2.5, median);  // (2 + 3) / 2
    }

    [TestMethod]
    public void PearsonCorrelation_WithPerfectCorrelation_ReturnsOne()
    {
        // Arrange
        var x = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var y = new[] { 2.0, 4.0, 6.0, 8.0, 10.0 };  // y = 2x

        // Act
        double correlation = StatisticsHelper.PearsonCorrelation(x, y);

        // Assert
        Assert.AreEqual(1.0, correlation, 1e-10);
    }

    [TestMethod]
    public void PearsonCorrelation_WithNoCorrelation_ReturnsNearZero()
    {
        // Arrange
        var x = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var y = new[] { 5.0, 3.0, 4.0, 1.0, 2.0 };  // Random order

        // Act
        double correlation = StatisticsHelper.PearsonCorrelation(x, y);

        // Assert
        Assert.IsTrue(Math.Abs(correlation) < 0.3,
            "Correlation should be close to 0");
    }
}
```

### Step 3: Create MatrixHelperTests.cs

```csharp
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AiDotNet.Helpers;
using AiDotNet.Numerics;

namespace AiDotNetTests.Helpers;

[TestClass]
public class MatrixHelperTests
{
    [TestMethod]
    public void Multiply_WithValidDimensions_ReturnsCorrectResult()
    {
        // Arrange
        var A = new Matrix<double>(2, 3);
        A[0, 0] = 1; A[0, 1] = 2; A[0, 2] = 3;
        A[1, 0] = 4; A[1, 1] = 5; A[1, 2] = 6;

        var B = new Matrix<double>(3, 2);
        B[0, 0] = 7; B[0, 1] = 8;
        B[1, 0] = 9; B[1, 1] = 10;
        B[2, 0] = 11; B[2, 1] = 12;

        // Act
        var C = MatrixHelper.Multiply(A, B);

        // Assert
        Assert.AreEqual(2, C.Rows);
        Assert.AreEqual(2, C.Columns);
        Assert.AreEqual(58.0, C[0, 0]);   // 1*7 + 2*9 + 3*11
        Assert.AreEqual(64.0, C[0, 1]);   // 1*8 + 2*10 + 3*12
        Assert.AreEqual(139.0, C[1, 0]);  // 4*7 + 5*9 + 6*11
        Assert.AreEqual(154.0, C[1, 1]);  // 4*8 + 5*10 + 6*12
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void Multiply_WithIncompatibleDimensions_ThrowsException()
    {
        // Arrange
        var A = new Matrix<double>(2, 3);
        var B = new Matrix<double>(2, 2);  // Incompatible: A.Columns != B.Rows

        // Act
        var C = MatrixHelper.Multiply(A, B);

        // Assert - Exception expected
    }

    [TestMethod]
    public void Transpose_CreatesCorrectTranspose()
    {
        // Arrange
        var A = new Matrix<double>(2, 3);
        A[0, 0] = 1; A[0, 1] = 2; A[0, 2] = 3;
        A[1, 0] = 4; A[1, 1] = 5; A[1, 2] = 6;

        // Act
        var AT = MatrixHelper.Transpose(A);

        // Assert
        Assert.AreEqual(3, AT.Rows);
        Assert.AreEqual(2, AT.Columns);
        Assert.AreEqual(1.0, AT[0, 0]);
        Assert.AreEqual(4.0, AT[0, 1]);
        Assert.AreEqual(6.0, AT[2, 1]);
    }

    [TestMethod]
    public void Determinant_ForIdentityMatrix_ReturnsOne()
    {
        // Arrange
        var I = Matrix<double>.Identity(3);

        // Act
        double det = MatrixHelper.Determinant(I);

        // Assert
        Assert.AreEqual(1.0, det, 1e-10);
    }

    [TestMethod]
    public void Determinant_ForSingularMatrix_ReturnsZero()
    {
        // Arrange - Rows are linearly dependent
        var A = new Matrix<double>(2, 2);
        A[0, 0] = 1; A[0, 1] = 2;
        A[1, 0] = 2; A[1, 1] = 4;  // Row 2 = 2 * Row 1

        // Act
        double det = MatrixHelper.Determinant(A);

        // Assert
        Assert.AreEqual(0.0, det, 1e-10);
    }
}
```

### Step 4: Create MatrixSolutionHelperTests.cs

```csharp
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AiDotNet.Helpers;
using AiDotNet.Numerics;

namespace AiDotNetTests.Helpers;

[TestClass]
public class MatrixSolutionHelperTests
{
    [TestMethod]
    public void Solve_WithSimpleSystem_ReturnsCorrectSolution()
    {
        // Arrange - Solve: 2x + y = 5, x + 3y = 8
        var A = new Matrix<double>(2, 2);
        A[0, 0] = 2; A[0, 1] = 1;
        A[1, 0] = 1; A[1, 1] = 3;

        var b = new Vector<double>(new[] { 5.0, 8.0 });

        // Act
        var x = MatrixSolutionHelper.Solve(A, b);

        // Assert - Solution should be x=1, y=3
        Assert.AreEqual(2, x.Length);
        Assert.AreEqual(1.0, x[0], 1e-10);
        Assert.AreEqual(3.0, x[1], 1e-10);

        // Verify: Ax = b
        var result = A.Multiply(x);
        Assert.AreEqual(5.0, result[0], 1e-10);
        Assert.AreEqual(8.0, result[1], 1e-10);
    }

    [TestMethod]
    public void Pseudoinverse_ForFullRankMatrix_MatchesInverse()
    {
        // Arrange
        var A = new Matrix<double>(2, 2);
        A[0, 0] = 1; A[0, 1] = 2;
        A[1, 0] = 3; A[1, 1] = 4;

        // Act
        var Aplus = MatrixSolutionHelper.Pseudoinverse(A);
        var Ainv = MatrixHelper.Inverse(A);

        // Assert - Pseudoinverse should match inverse for full-rank square matrix
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                Assert.AreEqual(Ainv[i, j], Aplus[i, j], 1e-10);
            }
        }
    }

    [TestMethod]
    public void Pseudoinverse_ForRankDeficientMatrix_ReturnsMoorePenrose()
    {
        // Arrange - Rank 1 matrix
        var A = new Matrix<double>(2, 2);
        A[0, 0] = 1; A[0, 1] = 2;
        A[1, 0] = 2; A[1, 1] = 4;  // Row 2 = 2 * Row 1

        // Act
        var Aplus = MatrixSolutionHelper.Pseudoinverse(A);

        // Assert - Should satisfy A * A+ * A = A
        var result = A.Multiply(Aplus).Multiply(A);
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                Assert.AreEqual(A[i, j], result[i, j], 1e-6);
            }
        }
    }

    [TestMethod]
    public void LeastSquares_WithOverdeterminedSystem_FindsBestFit()
    {
        // Arrange - Fit line y = mx + b to points
        // Points: (0,1), (1,3), (2,5), (3,7) -> y = 2x + 1
        var A = new Matrix<double>(4, 2);  // [1, x] for each point
        A[0, 0] = 1; A[0, 1] = 0;
        A[1, 0] = 1; A[1, 1] = 1;
        A[2, 0] = 1; A[2, 1] = 2;
        A[3, 0] = 1; A[3, 1] = 3;

        var b = new Vector<double>(new[] { 1.0, 3.0, 5.0, 7.0 });

        // Act
        var x = MatrixSolutionHelper.LeastSquares(A, b);

        // Assert - Should find slope=2, intercept=1
        Assert.AreEqual(2, x.Length);
        Assert.AreEqual(1.0, x[0], 1e-10);  // Intercept
        Assert.AreEqual(2.0, x[1], 1e-10);  // Slope
    }
}
```

---

## Complete Test Examples

### MathHelperTests.cs (Full Example)

```csharp
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AiDotNet.Helpers;

namespace AiDotNetTests.Helpers;

[TestClass]
public class MathHelperTests
{
    private const double Tolerance = 1e-10;

    #region Numeric Operations

    [TestMethod]
    public void GetNumericOperations_ForDouble_ReturnsValidOperations()
    {
        var ops = MathHelper.GetNumericOperations<double>();
        Assert.IsNotNull(ops);
        Assert.AreEqual(1.0, ops.One);
        Assert.AreEqual(0.0, ops.Zero);
    }

    [TestMethod]
    public void GetNumericOperations_ForFloat_ReturnsValidOperations()
    {
        var ops = MathHelper.GetNumericOperations<float>();
        Assert.IsNotNull(ops);
        Assert.AreEqual(1.0f, ops.One);
        Assert.AreEqual(0.0f, ops.Zero);
    }

    [TestMethod]
    public void GetNumericOperations_ForInt_ReturnsValidOperations()
    {
        var ops = MathHelper.GetNumericOperations<int>();
        Assert.IsNotNull(ops);
        Assert.AreEqual(1, ops.One);
        Assert.AreEqual(0, ops.Zero);
    }

    #endregion

    #region Clamp

    [TestMethod]
    public void Clamp_BelowMin_ReturnsMin()
    {
        double result = MathHelper.Clamp(5.0, 10.0, 20.0);
        Assert.AreEqual(10.0, result);
    }

    [TestMethod]
    public void Clamp_AboveMax_ReturnsMax()
    {
        double result = MathHelper.Clamp(25.0, 10.0, 20.0);
        Assert.AreEqual(20.0, result);
    }

    [TestMethod]
    public void Clamp_InRange_ReturnsValue()
    {
        double result = MathHelper.Clamp(15.0, 10.0, 20.0);
        Assert.AreEqual(15.0, result);
    }

    #endregion

    #region Sigmoid

    [TestMethod]
    public void Sigmoid_AtZero_ReturnsHalf()
    {
        double result = MathHelper.Sigmoid(0.0);
        Assert.AreEqual(0.5, result, Tolerance);
    }

    [TestMethod]
    public void Sigmoid_LargePositive_ApproachesOne()
    {
        double result = MathHelper.Sigmoid(10.0);
        Assert.IsTrue(result > 0.999);
    }

    [TestMethod]
    public void Sigmoid_LargeNegative_ApproachesZero()
    {
        double result = MathHelper.Sigmoid(-10.0);
        Assert.IsTrue(result < 0.001);
    }

    #endregion

    #region Special Functions

    [TestMethod]
    public void BesselI0_AtZero_ReturnsOne()
    {
        double result = MathHelper.BesselI0(0.0);
        Assert.AreEqual(1.0, result, Tolerance);
    }

    [TestMethod]
    public void Gamma_OfFive_Returns24()
    {
        double result = MathHelper.Gamma(5.0);
        Assert.AreEqual(24.0, result, 1e-6);
    }

    [TestMethod]
    public void Factorial_OfZero_ReturnsOne()
    {
        double result = MathHelper.Factorial<double>(0);
        Assert.AreEqual(1.0, result);
    }

    [TestMethod]
    public void Factorial_OfFive_Returns120()
    {
        double result = MathHelper.Factorial<double>(5);
        Assert.AreEqual(120.0, result);
    }

    #endregion

    #region Floating-Point Equality

    [TestMethod]
    public void AlmostEqual_Identical_ReturnsTrue()
    {
        Assert.IsTrue(MathHelper.AlmostEqual(1.0, 1.0));
    }

    [TestMethod]
    public void AlmostEqual_WithinTolerance_ReturnsTrue()
    {
        Assert.IsTrue(MathHelper.AlmostEqual(1.0, 1.0 + 1e-9));
    }

    [TestMethod]
    public void AlmostEqual_OutsideTolerance_ReturnsFalse()
    {
        Assert.IsFalse(MathHelper.AlmostEqual(1.0, 1.1, 0.01));
    }

    #endregion

    #region Trigonometry

    [TestMethod]
    public void Sin_AtZero_ReturnsZero()
    {
        double result = MathHelper.Sin(0.0);
        Assert.AreEqual(0.0, result, Tolerance);
    }

    [TestMethod]
    public void Cos_AtZero_ReturnsOne()
    {
        double result = MathHelper.Cos(0.0);
        Assert.AreEqual(1.0, result, Tolerance);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public void ArcSin_OutOfRange_ThrowsException()
    {
        MathHelper.ArcSin(2.0);
    }

    #endregion

    #region Min/Max

    [TestMethod]
    public void Min_ReturnsSmaller()
    {
        double result = MathHelper.Min(5.0, 10.0);
        Assert.AreEqual(5.0, result);
    }

    [TestMethod]
    public void Max_ReturnsLarger()
    {
        double result = MathHelper.Max(5.0, 10.0);
        Assert.AreEqual(10.0, result);
    }

    #endregion
}
```

---

## Success Criteria

### Definition of Done

- [ ] 4 test files created (Math, Statistics, Matrix, MatrixSolution)
- [ ] Minimum 30 tests for MathHelper (covers all major functions)
- [ ] Minimum 15 tests each for other helpers (45 total)
- [ ] All tests passing (0 failures)
- [ ] Code coverage >= 70% for each helper
- [ ] Edge cases tested (zero, negative, infinity, NaN)
- [ ] Numerical accuracy validated

### Quality Checklist

- [ ] All numeric types tested (double, float, decimal, int)
- [ ] Special functions compared to known values
- [ ] Matrix operations verify mathematical properties
- [ ] Linear system solutions verified with Ax = b
- [ ] Tolerance used for floating-point comparisons
- [ ] Exception cases tested (division by zero, out of range)

---

## Next Steps

After completing this issue:

1. **Run full test suite** and verify all pass
2. **Review code coverage** to identify gaps
3. **Performance test** complex math functions
4. **Move to Issue #350** (Model Helper tests)

---

**Happy Testing!** Accurate math is the foundation of reliable AI.
