# Issue #355: Junior Developer Implementation Guide
## 2D Interpolation and Specialized Methods - Unit Tests and Validation

---

## Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [Understanding 2D and Specialized Interpolation Methods](#understanding-2d-and-specialized-interpolation-methods)
3. [Current Implementation Status](#current-implementation-status)
4. [Testing Strategy](#testing-strategy)
5. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
6. [Common Pitfalls](#common-pitfalls)

---

## Understanding the Problem

### What Are We Solving?

The AiDotNet library has **14 specialized and 2D interpolation implementations** that currently have **0% test coverage**. These include advanced 2D methods (already have Bilinear and Bicubic), radial basis functions, signal processing methods, and specialized splines.

### Methods to Test (Issue #355)

**2D Interpolation** (3 methods):
1. **CubicConvolutionInterpolation** - Image resampling with cubic kernel
2. **LanczosInterpolation** - High-quality image resampling

**Radial Basis Functions** (3 methods):
3. **RadialBasisFunctionInterpolation** - Generic RBF interpolation
4. **ThinPlateSplineInterpolation** - Smooth surface fitting
5. **MultiquadricInterpolation** - RBF with multiquadric kernel
6. **ShepardsMethodInterpolation** - Inverse distance weighting

**Signal Processing** (3 methods):
7. **SincInterpolation** - Band-limited interpolation
8. **WhittakerShannonInterpolation** - Perfect reconstruction for bandlimited signals
9. **TrigonometricInterpolation** - Fourier-based interpolation

**Specialized Splines** (3 methods):
10. **NaturalSplineInterpolation** - Natural cubic spline (may duplicate CubicSpline)
11. **NotAKnotSplineInterpolation** - Not-a-knot boundary conditions
12. **PchipInterpolation** - Piecewise Cubic Hermite (may duplicate MonotoneCubic)

**Other** (2 methods):
13. **NewtonDividedDifferenceInterpolation** - Newton polynomial form
14. **BicubicInterpolation** - Already covered in Issue #352

---

## Understanding 2D and Specialized Interpolation Methods

### Category 1: Image Resampling Methods

These are optimized for image processing (upscaling, downscaling, rotation).

#### 1. Cubic Convolution Interpolation

**What it does**: Uses cubic kernel for image resampling.

**Key Formula** (1D version):
```
For distance d from pixel center:
  w(d) = { (a+2)|d|³ - (a+3)|d|² + 1           if |d| ≤ 1
         { a|d|³ - 5a|d|² + 8a|d| - 4a         if 1 < |d| < 2
         { 0                                    if |d| ≥ 2

where a = -0.5 (typical value)
```

**Use Cases**:
- Image upscaling
- Video processing
- Texture mapping

**Test Focus**:
```csharp
[Fact]
public void CubicConvolution_ImageUpscale_SmoothResult()
{
    // Arrange - 2x2 image
    var x = new Vector<double>(new[] { 0.0, 1.0 });
    var y = new Vector<double>(new[] { 0.0, 1.0 });
    var z = new Matrix<double>(2, 2);
    z[0, 0] = 0.0; z[0, 1] = 0.5;
    z[1, 0] = 0.5; z[1, 1] = 1.0;

    var interp = new CubicConvolutionInterpolation<double>(x, y, z);

    // Act - Upscale to 4x4 by sampling intermediate points
    for (double xi = 0.0; xi <= 1.0; xi += 0.25)
    {
        for (double yi = 0.0; yi <= 1.0; yi += 0.25)
        {
            double value = interp.Interpolate(xi, yi);

            // Assert - Should be smooth and within range
            Assert.True(value >= 0.0 && value <= 1.0);
            Assert.True(double.IsFinite(value));
        }
    }
}
```

#### 2. Lanczos Interpolation

**What it does**: High-quality image resampling using sinc function with windowing.

**Key Formula**:
```
L(x) = { sinc(x) * sinc(x/a)    if |x| < a
       { 0                       if |x| ≥ a

where a is the filter size (typically 2 or 3)
sinc(x) = sin(πx) / (πx)
```

**Properties**:
- Very high quality
- Slow (many multiplications)
- No overshoot
- Sharp edges preserved

**Use Cases**:
- Professional image editing (Photoshop, GIMP)
- Video upscaling
- When quality is more important than speed

**Test Focus**:
```csharp
[Fact]
public void Lanczos_StepFunction_PreservesSharpEdge()
{
    // Arrange - Step function (sharp edge)
    var x = new Vector<double>(new[] { 0.0, 0.5, 0.51, 1.0 });
    var y = new Vector<double>(new[] { 0.0, 1.0 });
    var z = new Matrix<double>(4, 2);
    // Left half = 0, right half = 1
    z[0, 0] = 0.0; z[0, 1] = 0.0;
    z[1, 0] = 0.0; z[1, 1] = 0.0;
    z[2, 0] = 1.0; z[2, 1] = 1.0;
    z[3, 0] = 1.0; z[3, 1] = 1.0;

    var lanczos = new LanczosInterpolation<double>(x, y, z);

    // Act - Sample near edge
    double nearEdge = lanczos.Interpolate(0.5, 0.5);

    // Assert - Should be close to edge (0.0 or 1.0), not blurred
    Assert.True(nearEdge < 0.3 || nearEdge > 0.7,
        "Edge should be preserved, not blurred");
}
```

### Category 2: Radial Basis Functions (RBF)

RBF methods use distance-based basis functions to interpolate scattered data.

**General Form**:
```
f(x) = Σ w_i * φ(||x - x_i||)

where:
  w_i = weights (computed from data)
  φ = radial basis function (depends on distance)
  ||x - x_i|| = distance from x to data point i
```

#### 3. Radial Basis Function Interpolation (Generic)

**Common Basis Functions**:
- **Gaussian**: φ(r) = exp(-r²/σ²)
- **Multiquadric**: φ(r) = sqrt(r² + c²)
- **Inverse Multiquadric**: φ(r) = 1/sqrt(r² + c²)
- **Thin Plate Spline**: φ(r) = r² * ln(r)

**Test Focus**:
```csharp
[Theory]
[InlineData("Gaussian")]
[InlineData("Multiquadric")]
[InlineData("InverseMultiquadric")]
[InlineData("ThinPlateSpline")]
public void RBF_WithDifferentBasisFunctions_Interpolates(string basisType)
{
    // Arrange - Scattered 2D points
    var (x, y, z) = CreateScattered2DData();

    var rbf = new RadialBasisFunctionInterpolation<double>(
        x, y, z, basisType);

    // Act & Assert - Test at known points
    for (int i = 0; i < x.Length; i++)
    {
        double result = rbf.Interpolate(x[i], y[i]);
        Assert.Equal(z[i], result, precision: 5);
    }
}
```

#### 4. Thin Plate Spline

**What it does**: Minimizes bending energy (like bending a thin metal plate).

**Basis Function**: φ(r) = r² * ln(r)

**Use Cases**:
- Image warping/morphing
- Medical image registration
- Cartography (map projections)
- Computer vision

**Property**: Exact interpolation at data points, smooth everywhere else.

**Test Focus**:
```csharp
[Fact]
public void ThinPlateSpline_AtDataPoints_ExactValues()
{
    // Arrange
    var x = new Vector<double>(new[] { 0.0, 1.0, 0.5, 0.25, 0.75 });
    var y = new Vector<double>(new[] { 0.0, 0.0, 1.0, 0.5, 0.5 });
    var z = new Vector<double>(new[] { 0.0, 1.0, 0.5, 0.3, 0.7 });

    var tps = new ThinPlateSplineInterpolation<double>(x, y, z);

    // Act & Assert - Exact at data points
    for (int i = 0; i < x.Length; i++)
    {
        double result = tps.Interpolate(x[i], y[i]);
        Assert.Equal(z[i], result, precision: 10);
    }
}
```

#### 5. Multiquadric Interpolation

**Basis Function**: φ(r) = sqrt(r² + c²)

**Parameter c**: Shape parameter (affects smoothness)

**Test Focus**:
```csharp
[Theory]
[InlineData(0.1)]
[InlineData(1.0)]
[InlineData(10.0)]
public void Multiquadric_DifferentShapeParameter_DifferentSmoothness(double c)
{
    // Test effect of shape parameter on interpolation
}
```

#### 6. Shepard's Method (Inverse Distance Weighting)

**What it does**: Weights points by inverse distance.

**Formula**:
```
         Σ w_i * f_i
f(x) = ─────────────
           Σ w_i

where w_i = 1 / ||x - x_i||^p  (p is power parameter, typically 2)
```

**Properties**:
- Simple
- Exact at data points
- C⁰ continuous (not smooth - has kinks at data points)

**Test Focus**:
```csharp
[Fact]
public void Shepards_NearDataPoint_DominatedByNearestValue()
{
    // Arrange
    var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
    var y = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
    var z = new Vector<double>(new[] { 10.0, 50.0, 30.0 });

    var shepard = new ShepardsMethodInterpolation<double>(x, y, z);

    // Act - Sample very close to middle point
    double result = shepard.Interpolate(1.001, 0.001);

    // Assert - Should be dominated by nearby value (50.0)
    Assert.True(Math.Abs(result - 50.0) < 5.0);
}
```

### Category 3: Signal Processing Methods

These are specialized for time-series and frequency-domain data.

#### 7. Sinc Interpolation

**What it does**: Uses sinc function (sin(x)/x) for interpolation.

**Formula**:
```
f(x) = Σ f_i * sinc((x - x_i) / h)

where sinc(x) = sin(πx) / (πx)
```

**Property**: Perfect for bandlimited signals (no frequencies above Nyquist).

**Use Cases**:
- Audio resampling
- Signal processing
- Digital communications

**Test Focus**:
```csharp
[Fact]
public void Sinc_BandlimitedSignal_PerfectReconstruction()
{
    // Arrange - Sample a sine wave at Nyquist rate
    int n = 20;
    double freq = 2.0;  // 2 Hz
    double sampleRate = 10.0;  // 10 samples/sec (well above Nyquist)

    var x = new Vector<double>(n);
    var y = new Vector<double>(n);

    for (int i = 0; i < n; i++)
    {
        x[i] = i / sampleRate;
        y[i] = Math.Sin(2 * Math.PI * freq * x[i]);
    }

    var sinc = new SincInterpolation<double>(x, y);

    // Act - Test at intermediate points
    for (double t = 0.0; t < 2.0; t += 0.01)
    {
        double interpolated = sinc.Interpolate(t);
        double expected = Math.Sin(2 * Math.PI * freq * t);

        // Assert - Should be very close (perfect reconstruction)
        Assert.Equal(expected, interpolated, precision: 3);
    }
}
```

#### 8. Whittaker-Shannon Interpolation

**What it does**: Theoretical perfect reconstruction of bandlimited signals.

**Formula** (same as sinc but with windowing):
```
f(t) = Σ f[n] * sinc((t - nT) / T)

where T is the sampling period
```

**Use Cases**:
- Digital signal processing theory
- Audio processing
- Telecommunications

**Test Focus**: Similar to sinc interpolation.

#### 9. Trigonometric Interpolation

**What it does**: Uses trigonometric polynomials (Fourier series).

**Formula**:
```
f(x) = a₀ + Σ [aₖ cos(kx) + bₖ sin(kx)]
```

**Property**: Periodic interpolation.

**Use Cases**:
- Periodic signals
- Fourier analysis
- Circular data

**Test Focus**:
```csharp
[Fact]
public void Trigonometric_PeriodicFunction_ExactReconstruction()
{
    // Arrange - Sample a periodic function
    int n = 10;
    var x = new Vector<double>(n);
    var y = new Vector<double>(n);

    for (int i = 0; i < n; i++)
    {
        x[i] = 2 * Math.PI * i / n;
        y[i] = Math.Sin(2 * x[i]) + 0.5 * Math.Cos(3 * x[i]);
    }

    var trig = new TrigonometricInterpolation<double>(x, y);

    // Act & Assert - Should reconstruct trigonometric function exactly
    for (double testX = 0.0; testX < 2 * Math.PI; testX += 0.1)
    {
        double interpolated = trig.Interpolate(testX);
        double expected = Math.Sin(2 * testX) + 0.5 * Math.Cos(3 * testX);

        Assert.Equal(expected, interpolated, precision: 8);
    }
}
```

### Category 4: Specialized Splines

#### 10. Natural Spline Interpolation

**Note**: May be duplicate of CubicSplineInterpolation. Check implementation.

**Boundary Conditions**: S''(x₀) = S''(xₙ) = 0

**Test Focus**: Same as CubicSplineInterpolation if duplicate.

#### 11. Not-a-Knot Spline Interpolation

**What it does**: Uses "not-a-knot" boundary conditions.

**Boundary Conditions**: Third derivative continuous at x₁ and xₙ₋₁.

**Effect**: Interior spline pieces extend smoothly to boundaries.

**Test Focus**:
```csharp
[Fact]
public void NotAKnot_ThirdDerivativeContinuous_AtSecondPoint()
{
    // Arrange
    var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
    var y = new Vector<double>(new[] { 0.0, 1.0, 0.5, 1.5, 1.0 });

    var spline = new NotAKnotSplineInterpolation<double>(x, y);

    // Act - Compute third derivative numerically at x[1]
    // (complex - may skip this test if too difficult)

    // Alternative: Test that it produces different results than natural spline
    var naturalSpline = new NaturalSplineInterpolation<double>(x, y);

    double notAKnotValue = spline.Interpolate(0.5);
    double naturalValue = naturalSpline.Interpolate(0.5);

    // Assert - Should differ
    Assert.NotEqual(notAKnotValue, naturalValue, precision: 5);
}
```

#### 12. PCHIP Interpolation

**Note**: May be duplicate of MonotoneCubicInterpolation. Check implementation.

**Full Name**: Piecewise Cubic Hermite Interpolating Polynomial

**Test Focus**: Same as MonotoneCubicInterpolation if duplicate.

### Category 5: Other Methods

#### 13. Newton Divided Difference Interpolation

**What it does**: Polynomial interpolation using Newton form.

**Advantage over Lagrange**: Easy to add new data points incrementally.

**Formula**:
```
P(x) = f[x₀] + f[x₀,x₁](x-x₀) + f[x₀,x₁,x₂](x-x₀)(x-x₁) + ...

where f[x₀,x₁,...,xₖ] are divided differences
```

**Use Cases**:
- When data arrives incrementally
- Teaching/education

**Test Focus**:
```csharp
[Fact]
public void NewtonDividedDifference_PolynomialData_ExactReconstruction()
{
    // Arrange - Cubic polynomial
    var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
    var y = new Vector<double>(5);

    for (int i = 0; i < 5; i++)
    {
        double xi = x[i];
        y[i] = xi * xi * xi - 2 * xi * xi + xi + 3;
    }

    var newton = new NewtonDividedDifferenceInterpolation<double>(x, y);

    // Act & Assert - Exact for polynomial
    for (double testX = 0.0; testX <= 4.0; testX += 0.25)
    {
        double predicted = newton.Interpolate(testX);
        double expected = testX * testX * testX - 2 * testX * testX + testX + 3;

        Assert.Equal(expected, predicted, precision: 8);
    }
}
```

---

## Current Implementation Status

### Existing Files

All in `C:\Users\cheat\source\repos\AiDotNet\src\Interpolation\`:

**2D Methods**:
- `CubicConvolutionInterpolation.cs`
- `LanczosInterpolation.cs`

**RBF Methods**:
- `RadialBasisFunctionInterpolation.cs`
- `ThinPlateSplineInterpolation.cs`
- `MultiquadricInterpolation.cs`
- `ShepardsMethodInterpolation.cs`

**Signal Processing**:
- `SincInterpolation.cs`
- `WhittakerShannonInterpolation.cs`
- `TrigonometricInterpolation.cs`

**Specialized Splines**:
- `NaturalSplineInterpolation.cs`
- `NotAKnotSplineInterpolation.cs`
- `PchipInterpolation.cs`

**Other**:
- `NewtonDividedDifferenceInterpolation.cs`

**Test Status**: 0% coverage

---

## Testing Strategy

### Test Categories

#### 1. Constructor and Basic Tests

Standard tests for all methods:
- Constructor validation
- Exact point tests
- Invalid input tests

#### 2. Method-Specific Tests

**Image Resampling** (Cubic Convolution, Lanczos):
- Test upscaling
- Test downscaling
- Test edge preservation
- Test smoothness

**RBF Methods**:
- Test scattered data
- Test exact interpolation at data points
- Test smoothness
- Test parameter effects (shape, power)

**Signal Processing**:
- Test bandlimited signals
- Test periodic functions
- Test Nyquist criterion
- Test aliasing behavior

**Specialized Splines**:
- Test boundary conditions
- Compare with standard cubic spline

#### 3. Performance Tests

Some methods (RBF, Lanczos) may be slow:
- Test with reasonable data sizes
- Document performance characteristics

---

## Step-by-Step Implementation Guide

### Phase 1: Set Up Test Infrastructure (1 hour)

Create 14 test files (one per method).

### Phase 2: Implement Tests (12-18 hours)

**Estimated time per method**: 1-1.5 hours each

**Priority Order**:
1. **Cubic Convolution, Lanczos** (2D, important for images)
2. **Thin Plate Spline, Multiquadric, Shepard** (RBF, widely used)
3. **Newton Divided Difference** (educational value)
4. **Sinc, Whittaker-Shannon** (signal processing)
5. **Not-a-Knot, PCHIP** (spline variants)
6. **Trigonometric** (periodic signals)
7. **Generic RBF** (if different from specific RBF types)
8. **Natural Spline** (if different from Cubic Spline)

### Phase 3: Run and Debug (2-3 hours)

```bash
dotnet test tests/AiDotNetTests.csproj --filter "FullyQualifiedName~Interpolation"
```

### Phase 4: Documentation (30 minutes)

---

## Common Pitfalls

### Pitfall 1: Sinc Function at Zero

sinc(0) = 1 (not 0/0). Handle specially:
```csharp
double Sinc(double x)
{
    if (Math.Abs(x) < 1e-10)
        return 1.0;
    return Math.Sin(Math.PI * x) / (Math.PI * x);
}
```

### Pitfall 2: RBF Ill-Conditioning

RBF matrices can be ill-conditioned (nearly singular).

**Test with regularization** (if implementation supports it).

### Pitfall 3: Lanczos Filter Size

Lanczos-2 vs Lanczos-3 produces different results.

**Document which is used**.

### Pitfall 4: Periodic vs Non-Periodic

Trigonometric interpolation assumes periodic data.

**Test with periodic boundary conditions**.

### Pitfall 5: Duplicate Implementations

PCHIP may == MonotoneCubic, NaturalSpline may == CubicSpline.

**Check implementations** - if duplicates, reference existing tests.

---

## Checklist Summary

### Phase 1: Setup (1 hour)
- [ ] Create 14 test files
- [ ] Verify compilation

### Phase 2: Implement Tests (12-18 hours)
- [ ] CubicConvolutionInterpolation
- [ ] LanczosInterpolation
- [ ] RadialBasisFunctionInterpolation
- [ ] ThinPlateSplineInterpolation
- [ ] MultiquadricInterpolation
- [ ] ShepardsMethodInterpolation
- [ ] SincInterpolation
- [ ] WhittakerShannonInterpolation
- [ ] TrigonometricInterpolation
- [ ] NaturalSplineInterpolation
- [ ] NotAKnotSplineInterpolation
- [ ] PchipInterpolation
- [ ] NewtonDividedDifferenceInterpolation

### Phase 3: Validation (2-3 hours)
- [ ] All tests pass
- [ ] Coverage 75%+

### Phase 4: Documentation (30 minutes)
- [ ] Document test strategies

### Total Estimated Time: 15-22 hours

---

## Success Criteria

1. **All Tests Pass**: 100% pass rate
2. **Coverage**: 75%+ for each method
3. **Method-Specific Properties**: Verified for each type
4. **Documentation**: Test strategies documented

---

## Resources

- **RBF**: https://en.wikipedia.org/wiki/Radial_basis_function
- **Thin Plate Splines**: https://en.wikipedia.org/wiki/Thin_plate_spline
- **Sinc Interpolation**: https://en.wikipedia.org/wiki/Whittaker%E2%80%93Shannon_interpolation_formula
- **Lanczos**: https://en.wikipedia.org/wiki/Lanczos_resampling
- **Newton Divided Differences**: https://en.wikipedia.org/wiki/Newton_polynomial
- **Shepard's Method**: https://en.wikipedia.org/wiki/Inverse_distance_weighting
