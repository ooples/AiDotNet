# Issue #365: Junior Developer Implementation Guide
## Implement Tests for Advanced Wavelets

---

## Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [Understanding Advanced Wavelets](#understanding-advanced-wavelets)
3. [Files Requiring Tests](#files-requiring-tests)
4. [Testing Strategy](#testing-strategy)
5. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
6. [Expected Test Structure](#expected-test-structure)
7. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)

---

## Understanding the Problem

### What Are We Solving?

Advanced wavelet functions in `src/WaveletFunctions/` currently have **0% test coverage**. We need to achieve **80%+ test coverage** for these specialized wavelets used in signal processing, image analysis, and scientific computing.

### Why Testing Matters for Advanced Wavelets

Advanced wavelets serve specialized purposes:
- **Meyer**: Frequency domain analysis, spectral processing
- **Shannon**: Perfect frequency band isolation
- **ComplexMorlet**: Time-frequency analysis (EEG, audio)
- **ComplexGaussian**: Multiscale edge detection
- **Gabor**: Texture analysis, computer vision
- **Paul**: Oscillatory signal analysis
- **DOG (Difference of Gaussians)**: Blob detection, feature extraction
- **FejérKorovkin**: Approximation theory, numerical methods
- **BattleLemarie**: Finite element analysis, wavelet-Galerkin methods

**Without tests**, we have no guarantee that:
- Complex mathematical formulas are correctly implemented
- Parameters behave as expected
- Frequency domain operations work correctly
- Edge cases are handled properly
- Integration with FFT is correct

---

## Understanding Advanced Wavelets

### Categories of Advanced Wavelets

#### 1. Frequency Domain Wavelets
**Examples**: Meyer, Shannon
**Characteristics**:
- Defined primarily in frequency domain
- Require Fourier transforms
- Excellent frequency localization
- Often infinitely differentiable

**Testing focus**: FFT operations, frequency band properties

#### 2. Complex Wavelets
**Examples**: ComplexMorlet, ComplexGaussian, Paul
**Characteristics**:
- Have real and imaginary components
- Provide phase information
- Excellent for oscillatory signals
- Used in time-frequency analysis

**Testing focus**: Complex arithmetic, phase properties, admissibility

#### 3. Texture Analysis Wavelets
**Examples**: Gabor
**Characteristics**:
- Tuned for specific orientations and frequencies
- Used in computer vision
- Multiple parameter dimensions

**Testing focus**: Orientation selectivity, frequency tuning, texture detection

#### 4. Edge Detection Wavelets
**Examples**: DOG (Difference of Gaussians), ComplexGaussian
**Characteristics**:
- Derivatives of smooth functions
- Scale-space analysis
- Blob and edge detection

**Testing focus**: Zero-crossings, scale properties, derivative orders

#### 5. Specialized Mathematical Wavelets
**Examples**: FejérKorovkin, BattleLemarie
**Characteristics**:
- Mathematical properties (polynomial reproduction)
- Numerical analysis applications
- Specific orthogonality or approximation properties

**Testing focus**: Mathematical properties, accuracy, orthogonality

---

## Files Requiring Tests

### 1. MeyerWavelet.cs
**Type**: Frequency domain, infinitely differentiable
**Complexity**: High (uses FFT)
**Key Properties**:
- Defined in frequency domain
- Compact support in frequency domain
- Infinitely smooth in time domain
- Band-limited

**Test Priority**: HIGH (complex implementation)

### 2. ShannonWavelet.cs
**Type**: Frequency domain, sinc-based
**Complexity**: High
**Key Properties**:
- Perfect frequency band isolation
- Sinc function in time domain
- Infinite support in time domain
- Ideal low-pass/high-pass separation

**Test Priority**: HIGH (theoretical importance)

### 3. ComplexMorletWavelet.cs
**Type**: Complex, Gaussian-modulated
**Complexity**: Moderate-High
**Key Properties**:
- Complex-valued (has real and imaginary parts)
- Gaussian envelope
- Central frequency parameter
- Excellent time-frequency localization

**Test Priority**: CRITICAL (widely used)

### 4. ComplexGaussianWavelet.cs
**Type**: Complex, derivative-based
**Complexity**: Moderate-High
**Key Properties**:
- Derivatives of Gaussian
- Complex-valued
- Order parameter
- Good for edge detection

**Test Priority**: HIGH

### 5. GaborWavelet.cs
**Type**: Real/Complex, oriented
**Complexity**: Moderate-High
**Key Properties**:
- Orientation parameter
- Frequency parameter
- Gaussian envelope
- Used in texture analysis

**Test Priority**: HIGH (computer vision applications)

### 6. PaulWavelet.cs
**Type**: Complex, analytic
**Complexity**: Moderate
**Key Properties**:
- Order parameter
- Good for oscillatory signals
- Analytic wavelet (complex)

**Test Priority**: MEDIUM

### 7. DOGWavelet.cs (Difference of Gaussians)
**Type**: Real, edge/blob detector
**Complexity**: Moderate
**Key Properties**:
- Two sigma parameters
- Approximates Laplacian of Gaussian
- Scale-space analysis

**Test Priority**: HIGH (computer vision)

### 8. FejérKorovkinWavelet.cs
**Type**: Real, polynomial approximation
**Complexity**: High
**Key Properties**:
- Polynomial reproduction
- Order parameter
- Approximation theory applications

**Test Priority**: MEDIUM

### 9. BattleLemarieWavelet.cs
**Type**: Real, orthogonal spline
**Complexity**: High
**Key Properties**:
- Spline-based
- Orthogonality
- Numerical analysis applications

**Test Priority**: MEDIUM

---

## Testing Strategy

### Test Categories for Advanced Wavelets

#### Category 1: Constructor and Parameter Validation

```csharp
// Meyer Wavelet - No parameters
[Fact]
public void MeyerWavelet_Constructor_CreatesValidInstance()
{
    var wavelet = new MeyerWavelet<double>();
    Assert.NotNull(wavelet);
}

// ComplexMorlet - Omega and Sigma parameters
[Theory]
[InlineData(5.0, 1.0)]   // Default
[InlineData(3.0, 0.5)]   // High frequency, narrow
[InlineData(10.0, 2.0)]  // Low frequency, wide
public void ComplexMorlet_Constructor_ValidParameters_CreatesValidInstance(double omega, double sigma)
{
    var wavelet = new ComplexMorletWavelet<double>(omega, sigma);
    Assert.NotNull(wavelet);
}

// Gabor - Multiple parameters
[Theory]
[InlineData(0.0, 5.0, 1.0, 0.0)]      // Horizontal orientation
[InlineData(Math.PI/2, 5.0, 1.0, 0.0)] // Vertical orientation
public void Gabor_Constructor_ValidParameters_CreatesValidInstance(double theta, double omega, double sigma, double psi)
{
    var wavelet = new GaborWavelet<double>(theta, omega, sigma, psi);
    Assert.NotNull(wavelet);
}

// DOG - Two sigmas
[Theory]
[InlineData(1.0, 2.0)]   // sigma2 > sigma1
[InlineData(0.5, 1.0)]
public void DOG_Constructor_ValidSigmas_CreatesValidInstance(double sigma1, double sigma2)
{
    var wavelet = new DOGWavelet<double>(sigma1, sigma2);
    Assert.NotNull(wavelet);
}

[Fact]
public void DOG_Constructor_Sigma2LessThanSigma1_ThrowsArgumentException()
{
    Assert.Throws<ArgumentException>(() => new DOGWavelet<double>(2.0, 1.0));
}
```

#### Category 2: Mathematical Correctness Tests

**For Real-Valued Wavelets (Meyer, Shannon, DOG, etc.)**:
```csharp
[Theory]
[InlineData(0.0)]
[InlineData(1.0)]
[InlineData(-1.0)]
public void Meyer_Calculate_KnownPoints_ReturnsFiniteValue(double x)
{
    var wavelet = new MeyerWavelet<double>();
    double value = wavelet.Calculate(x);

    Assert.False(double.IsNaN(value));
    Assert.False(double.IsInfinity(value));
}

[Fact]
public void Shannon_Calculate_CenterPoint_ReturnsMaxValue()
{
    var wavelet = new ShannonWavelet<double>();
    double valueAtZero = wavelet.Calculate(0.0);

    // Shannon wavelet (sinc) has maximum at x=0
    Assert.True(valueAtZero > 0.5);  // Should be close to 1
}

[Fact]
public void DOG_Calculate_AtOrigin_ReturnsExpectedValue()
{
    var wavelet = new DOGWavelet<double>(sigma1: 1.0, sigma2: 2.0);
    double value = wavelet.Calculate(0.0);

    // At x=0: DOG(0) = 1/(sqrt(2pi)) * (1/sigma1 - 1/sigma2)
    double expected = 1.0 / Math.Sqrt(2.0 * Math.PI) * (1.0 - 0.5);
    Assert.Equal(expected, value, precision: 6);
}
```

**For Complex-Valued Wavelets**:
```csharp
[Theory]
[InlineData(0.0, 0.0)]
[InlineData(1.0, 0.0)]
[InlineData(0.0, 1.0)]
public void ComplexMorlet_Calculate_KnownPoints_ReturnsFiniteValue(double real, double imag)
{
    var wavelet = new ComplexMorletWavelet<double>(omega: 5.0, sigma: 1.0);
    var z = new Complex<double>(real, imag);
    var result = wavelet.Calculate(z);

    Assert.False(double.IsNaN(result.Real));
    Assert.False(double.IsNaN(result.Imaginary));
    Assert.False(double.IsInfinity(result.Real));
    Assert.False(double.IsInfinity(result.Imaginary));
}

[Fact]
public void ComplexMorlet_Calculate_AtOrigin_HasExpectedMagnitude()
{
    var wavelet = new ComplexMorletWavelet<double>(omega: 5.0, sigma: 1.0);
    var z = new Complex<double>(0.0, 0.0);
    var result = wavelet.Calculate(z);

    double magnitude = Math.Sqrt(result.Real * result.Real + result.Imaginary * result.Imaginary);
    Assert.True(magnitude > 0.5);  // Should be close to 1 at origin
}
```

#### Category 3: Admissibility Tests (for Continuous Wavelets)

```csharp
[Fact]
public void ComplexMorlet_SatisfiesAdmissibilityCondition()
{
    // Admissibility: integral of wavelet over all x must be zero
    // For complex wavelets, integral of real part should be zero
    var wavelet = new ComplexMorletWavelet<double>(omega: 5.0, sigma: 1.0);

    double integralReal = 0.0;
    double step = 0.1;

    for (double x = -10.0; x <= 10.0; x += step)
    {
        var z = new Complex<double>(x, 0.0);
        var value = wavelet.Calculate(z);
        integralReal += value.Real * step;
    }

    Assert.Equal(0.0, integralReal, precision: 1);  // Should be close to zero
}

[Fact]
public void DOG_IntegralIsZero()
{
    var wavelet = new DOGWavelet<double>(1.0, 2.0);

    double integral = 0.0;
    double step = 0.05;

    for (double x = -20.0; x <= 20.0; x += step)
    {
        integral += wavelet.Calculate(x) * step;
    }

    Assert.Equal(0.0, integral, precision: 1);
}
```

#### Category 4: Symmetry and Antisymmetry Tests

```csharp
[Theory]
[InlineData(1.0)]
[InlineData(2.0)]
[InlineData(0.5)]
public void DOG_IsSymmetric(double x)
{
    var wavelet = new DOGWavelet<double>(1.0, 2.0);

    double positiveValue = wavelet.Calculate(x);
    double negativeValue = wavelet.Calculate(-x);

    Assert.Equal(positiveValue, negativeValue, precision: 10);
}

[Theory]
[InlineData(1.0)]
[InlineData(2.0)]
public void Paul_RealPart_IsAntisymmetric(double x)
{
    var wavelet = new PaulWavelet<double>(order: 4);

    var positiveValue = wavelet.Calculate(new Complex<double>(x, 0.0));
    var negativeValue = wavelet.Calculate(new Complex<double>(-x, 0.0));

    // Real part should be antisymmetric
    Assert.Equal(-positiveValue.Real, negativeValue.Real, precision: 10);
}
```

#### Category 5: Frequency Domain Properties (for Meyer, Shannon)

```csharp
[Fact]
public void Meyer_FrequencyDomain_HasCompactSupport()
{
    var wavelet = new MeyerWavelet<double>();
    var signal = CreateImpulse(256);

    // Decompose to get frequency domain representation
    var (approx, detail) = wavelet.Decompose(signal);

    // Meyer wavelet has compact support in frequency domain
    // Details should be zero outside specific frequency bands
    Assert.NotNull(approx);
    Assert.NotNull(detail);
}

[Fact]
public void Shannon_PerfectBandSeparation()
{
    var wavelet = new ShannonWavelet<double>();

    // Shannon wavelet provides perfect separation of frequency bands
    // Test with signal containing known frequencies
    var signal = CreateSinusoid(frequency: 0.25, length: 256);
    var (approx, detail) = wavelet.Decompose(signal);

    // Low frequencies should be in approximation
    // High frequencies should be in detail
}

private Vector<double> CreateImpulse(int length)
{
    var data = new double[length];
    data[length / 2] = 1.0;
    return new Vector<double>(data);
}

private Vector<double> CreateSinusoid(double frequency, int length)
{
    var data = new double[length];
    for (int i = 0; i < length; i++)
    {
        data[i] = Math.Sin(2.0 * Math.PI * frequency * i);
    }
    return new Vector<double>(data);
}
```

#### Category 6: Scale and Parameter Sensitivity Tests

```csharp
[Fact]
public void ComplexMorlet_IncreasingSigma_IncreasesWidth()
{
    var wavelet1 = new ComplexMorletWavelet<double>(omega: 5.0, sigma: 1.0);
    var wavelet2 = new ComplexMorletWavelet<double>(omega: 5.0, sigma: 2.0);

    // Calculate FWHM (Full Width at Half Maximum) for both
    double fwhm1 = MeasureWidthAtHalfMax(wavelet1);
    double fwhm2 = MeasureWidthAtHalfMax(wavelet2);

    Assert.True(fwhm2 > fwhm1);  // Larger sigma should give wider wavelet
}

[Fact]
public void Gabor_OrientationParameter_RotatesWavelet()
{
    var horizontal = new GaborWavelet<double>(theta: 0.0, omega: 5.0, sigma: 1.0, psi: 0.0);
    var vertical = new GaborWavelet<double>(theta: Math.PI/2, omega: 5.0, sigma: 1.0, psi: 0.0);

    // Test at symmetric point
    double valH = horizontal.Calculate(1.0);
    double valV = vertical.Calculate(1.0);

    // Values should differ due to orientation
    Assert.NotEqual(valH, valV, precision: 5);
}

private double MeasureWidthAtHalfMax(ComplexMorletWavelet<double> wavelet)
{
    // Find points where magnitude drops to half of maximum
    // (Implementation details)
    return 1.0;  // Placeholder
}
```

#### Category 7: Decomposition Tests

```csharp
[Fact]
public void Meyer_Decompose_ProducesSameLengthOutputs()
{
    var wavelet = new MeyerWavelet<double>();
    var signal = new Vector<double>(256);  // Must be power of 2 for FFT

    for (int i = 0; i < signal.Length; i++)
        signal[i] = Math.Sin(2.0 * Math.PI * i / 16.0);

    var (approx, detail) = wavelet.Decompose(signal);

    // Frequency domain wavelets might not downsample
    Assert.Equal(signal.Length / 2, approx.Length);
    Assert.Equal(signal.Length / 2, detail.Length);
}

[Fact]
public void ComplexMorlet_Decompose_ProducesComplexCoefficients()
{
    var wavelet = new ComplexMorletWavelet<double>(omega: 5.0, sigma: 1.0);
    var signal = CreateComplexSignal(128);

    var (approx, detail) = wavelet.Decompose(signal);

    Assert.NotNull(approx);
    Assert.NotNull(detail);
    Assert.True(approx.Length > 0);
    Assert.True(detail.Length > 0);
}

private Vector<Complex<double>> CreateComplexSignal(int length)
{
    var data = new Complex<double>[length];
    for (int i = 0; i < length; i++)
    {
        double angle = 2.0 * Math.PI * i / length;
        data[i] = new Complex<double>(Math.Cos(angle), Math.Sin(angle));
    }
    return new Vector<Complex<double>>(data);
}
```

#### Category 8: Coefficient Tests

```csharp
[Fact]
public void BattleLemarie_GetScalingCoefficients_ReturnsCorrectLength()
{
    var wavelet = new BattleLemarieWavelet<double>();
    var coeffs = wavelet.GetScalingCoefficients();

    Assert.NotNull(coeffs);
    Assert.True(coeffs.Length > 0);
}

[Fact]
public void FejérKorovkin_GetWaveletCoefficients_Normalized()
{
    var wavelet = new FejérKorovkinWavelet<double>();
    var coeffs = wavelet.GetWaveletCoefficients();

    double sumOfSquares = 0.0;
    for (int i = 0; i < coeffs.Length; i++)
    {
        sumOfSquares += coeffs[i] * coeffs[i];
    }

    // Should be normalized (or check specific normalization condition)
    Assert.True(sumOfSquares > 0.0);
}
```

#### Category 9: Edge Case Tests

```csharp
[Fact]
public void Meyer_Calculate_VeryLargeValue_ReturnsZero()
{
    var wavelet = new MeyerWavelet<double>();
    double value = wavelet.Calculate(1000.0);

    Assert.Equal(0.0, value, precision: 10);
}

[Fact]
public void ComplexMorlet_OmegaSigmaProduct_BelowThreshold_ThrowsException()
{
    // Admissibility requires omega * sigma > 5
    Assert.Throws<ArgumentException>(() =>
        new ComplexMorletWavelet<double>(omega: 2.0, sigma: 1.0));
}

[Fact]
public void Gabor_NegativeSigma_ThrowsArgumentException()
{
    Assert.Throws<ArgumentException>(() =>
        new GaborWavelet<double>(theta: 0.0, omega: 5.0, sigma: -1.0, psi: 0.0));
}
```

#### Category 10: Specific Wavelet Properties

**Paul Wavelet - Order Parameter**:
```csharp
[Theory]
[InlineData(1)]
[InlineData(2)]
[InlineData(4)]
public void Paul_DifferentOrders_ProduceDifferentShapes(int order)
{
    var wavelet = new PaulWavelet<double>(order);

    var value1 = wavelet.Calculate(new Complex<double>(1.0, 0.0));
    var value2 = wavelet.Calculate(new Complex<double>(2.0, 0.0));

    // Higher orders decay faster
    Assert.True(Math.Abs(value2.Real) < Math.Abs(value1.Real));
}
```

**DOG - Scale Ratio**:
```csharp
[Fact]
public void DOG_ApproximatesLoG_WhenSigmaRatioIs1_6()
{
    // DOG with sigma2/sigma1 ≈ 1.6 approximates Laplacian of Gaussian
    var dog = new DOGWavelet<double>(sigma1: 1.0, sigma2: 1.6);

    // Test Laplacian property: zero crossing at certain radius
    double innerValue = dog.Calculate(0.5);
    double outerValue = dog.Calculate(2.0);

    // Should have opposite signs (zero crossing between)
    Assert.True(innerValue * outerValue < 0);
}
```

**FejérKorovkin - Polynomial Reproduction**:
```csharp
[Fact]
public void FejérKorovkin_ReproducesPolynomials()
{
    var wavelet = new FejérKorovkinWavelet<double>();

    // Detail coefficients should be zero for polynomial signals
    var polynomial = CreatePolynomialSignal(degree: 2, length: 16);
    var (approx, detail) = wavelet.Decompose(polynomial);

    // Detail should be small (near zero) for polynomials
    double detailEnergy = CalculateEnergy(detail);
    Assert.True(detailEnergy < 0.01);
}

private Vector<double> CreatePolynomialSignal(int degree, int length)
{
    var data = new double[length];
    for (int i = 0; i < length; i++)
    {
        double x = (double)i / length;
        data[i] = Math.Pow(x, degree);
    }
    return new Vector<double>(data);
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

---

## Step-by-Step Implementation Guide

### Phase 1: Prioritize Wavelets by Usage

**High Priority** (Start Here):
1. ComplexMorletWavelet - Most widely used
2. GaborWavelet - Computer vision applications
3. DOGWavelet - Feature detection
4. MeyerWavelet - Frequency domain

**Medium Priority**:
5. ComplexGaussianWavelet
6. ShannonWavelet
7. PaulWavelet

**Lower Priority**:
8. FejérKorovkinWavelet
9. BattleLemarieWavelet

### Phase 2: Create Test Structure

```
tests/
  WaveletFunctions/
    AdvancedWavelets/
      ComplexWavelets/
        ComplexMorletWaveletTests.cs
        ComplexGaussianWaveletTests.cs
        PaulWaveletTests.cs
      FrequencyDomainWavelets/
        MeyerWaveletTests.cs
        ShannonWaveletTests.cs
      FeatureDetection/
        DOGWaveletTests.cs
        GaborWaveletTests.cs
      Specialized/
        FejérKorovkinWaveletTests.cs
        BattleLemarieWaveletTests.cs
```

### Phase 3: Implement ComplexMorlet Tests (Start Here)

```csharp
public class ComplexMorletWaveletTests
{
    [Fact]
    public void Constructor_DefaultParameters_CreatesValidInstance()
    {
        var wavelet = new ComplexMorletWavelet<double>();
        Assert.NotNull(wavelet);
    }

    [Theory]
    [InlineData(5.0, 1.0)]
    [InlineData(6.0, 1.5)]
    public void Constructor_ValidParameters_CreatesValidInstance(double omega, double sigma)
    {
        var wavelet = new ComplexMorletWavelet<double>(omega, sigma);
        Assert.NotNull(wavelet);
    }

    [Fact]
    public void Calculate_AtOrigin_ReturnsExpectedValue()
    {
        var wavelet = new ComplexMorletWavelet<double>(omega: 5.0, sigma: 1.0);
        var result = wavelet.Calculate(new Complex<double>(0.0, 0.0));

        // At origin: exp(0) = 1
        Assert.Equal(1.0, result.Real, precision: 6);
        Assert.Equal(0.0, result.Imaginary, precision: 6);
    }

    [Fact]
    public void Admissibility_OmegaSigmaProduct_AboveThreshold()
    {
        var wavelet = new ComplexMorletWavelet<double>(omega: 5.0, sigma: 1.0);

        // Test admissibility numerically
        // (Implementation)
    }

    // ... more tests
}
```

### Phase 4: Continue with Other Wavelets

Follow similar patterns for each wavelet, adapting tests to specific properties.

### Phase 5: Run and Validate

```bash
dotnet test --filter "FullyQualifiedName~AdvancedWavelets"
dotnet test /p:CollectCoverage=true
```

---

## Expected Test Structure

### ComplexMorletWaveletTests.cs (Complete Outline)
```csharp
public class ComplexMorletWaveletTests
{
    #region Constructor Tests
    [Fact] public void Constructor_DefaultParameters_CreatesValidInstance() { }
    [Theory] public void Constructor_ValidParameters_CreatesValidInstance(double omega, double sigma) { }
    [Fact] public void Constructor_InvalidOmegaSigmaProduct_ThrowsException() { }
    #endregion

    #region Calculate Tests
    [Fact] public void Calculate_AtOrigin_ReturnsExpectedValue() { }
    [Theory] public void Calculate_VariousPoints_ReturnsFiniteValues(double real, double imag) { }
    [Fact] public void Calculate_FarFromOrigin_ApproachesZero() { }
    #endregion

    #region Admissibility Tests
    [Fact] public void Admissibility_RealPartIntegral_IsZero() { }
    [Fact] public void Admissibility_ImaginaryPartIntegral_IsZero() { }
    #endregion

    #region Decomposition Tests
    [Fact] public void Decompose_ComplexSignal_ProducesValidCoefficients() { }
    [Fact] public void Decompose_DownsampledCorrectly() { }
    #endregion

    #region Coefficient Tests
    [Fact] public void GetScalingCoefficients_ReturnsValidLength() { }
    [Fact] public void GetWaveletCoefficients_Normalized() { }
    #endregion

    #region Scale Tests
    [Fact] public void IncreasingSigma_IncreasesWidth() { }
    [Fact] public void IncreasingOmega_IncreasesFrequency() { }
    #endregion
}
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Complex Number Arithmetic
**Problem**: Errors in complex multiplication, division.

**Solution**:
- Use library Complex type consistently
- Test real and imaginary parts separately
- Verify magnitude and phase independently

### Pitfall 2: FFT Requirements
**Problem**: Meyer/Shannon require specific signal lengths (powers of 2).

**Solution**:
```csharp
[Fact]
public void Meyer_Decompose_NonPowerOf2Length_ThrowsException()
{
    var wavelet = new MeyerWavelet<double>();
    var signal = new Vector<double>(100);  // Not a power of 2

    Assert.Throws<ArgumentException>(() => wavelet.Decompose(signal));
}
```

### Pitfall 3: Admissibility Numerical Integration
**Problem**: Numerical integration is approximate.

**Solution**:
- Use fine step size (0.01 or smaller)
- Extend integration range sufficiently (-20 to +20)
- Accept looser precision (1-2 decimal places)

### Pitfall 4: Parameter Constraints
**Problem**: Not all parameter combinations are valid.

**Solution**:
```csharp
// ComplexMorlet: omega * sigma >= 5 for admissibility
[Fact]
public void ComplexMorlet_OmegaSigmaProduct_BelowThreshold_Invalid()
{
    Assert.Throws<ArgumentException>(() =>
        new ComplexMorletWavelet<double>(omega: 2.0, sigma: 2.0));
}

// DOG: sigma2 > sigma1
[Fact]
public void DOG_Sigma2LessThanSigma1_Invalid()
{
    Assert.Throws<ArgumentException>(() =>
        new DOGWavelet<double>(sigma1: 2.0, sigma2: 1.0));
}
```

### Pitfall 5: Infinite Support Wavelets
**Problem**: Shannon, Gabor have infinite support (don't go to zero).

**Solution**:
- Test decay rate instead of zero value
- Use sufficiently large range for calculations
- Accept non-zero values at boundaries

```csharp
[Fact]
public void Shannon_Calculate_DecaysAtLargeX()
{
    var wavelet = new ShannonWavelet<double>();

    double value10 = Math.Abs(wavelet.Calculate(10.0));
    double value20 = Math.Abs(wavelet.Calculate(20.0));

    Assert.True(value20 < value10);  // Should decay, even if not to zero
}
```

### Pitfall 6: Not Testing Frequency Domain Properties
**Problem**: Ignoring frequency domain behavior for Meyer/Shannon.

**Solution**: Create frequency domain tests using FFT.

---

## References

### Mathematical References
1. **Mallat, S.**: "A Wavelet Tour of Signal Processing" - Comprehensive coverage
2. **Daubechies, I.**: "Ten Lectures on Wavelets" - Theoretical foundation
3. **Torrence, C. & Compo, G.P.**: "A Practical Guide to Wavelet Analysis" - Complex Morlet, Paul

### Online Resources
1. **PyWavelets**: Reference implementation for complex wavelets
2. **MATLAB Wavelet Toolbox**: Documentation for all wavelets
3. **Wikipedia - Continuous Wavelet Transform**: Good overview

### Specific Wavelet References
1. **Gabor Wavelets**: Daugman, J.G., "Complete discrete 2-D Gabor transforms"
2. **DOG**: Marr-Hildreth edge detection theory
3. **Meyer Wavelet**: Meyer, Y., "Wavelets and Operators"

---

## Success Criteria

Your implementation is complete when:

1. All 9 advanced wavelet classes have test files
2. Each file has 15-25 tests covering relevant categories
3. Code coverage is 80%+ for all files
4. Complex arithmetic is correctly tested
5. Frequency domain properties are validated
6. Parameter constraints are enforced
7. Admissibility is verified for continuous wavelets
8. All tests pass consistently
9. Edge cases and error conditions are covered

Good luck!
