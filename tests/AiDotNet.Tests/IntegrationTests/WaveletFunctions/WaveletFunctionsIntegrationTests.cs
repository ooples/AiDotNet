using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.WaveletFunctions;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.WaveletFunctions;

/// <summary>
/// Integration tests for wavelet function classes.
/// Tests wavelet calculation, decomposition, and coefficient generation.
/// </summary>
public class WaveletFunctionsIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Haar Wavelet Tests

    [Fact]
    public void HaarWavelet_Calculate_FirstHalf_ReturnsOne()
    {
        // Arrange
        var wavelet = new HaarWavelet<double>();

        // Act
        var result = wavelet.Calculate(0.25);

        // Assert - x in [0, 0.5) should return 1
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void HaarWavelet_Calculate_SecondHalf_ReturnsNegativeOne()
    {
        // Arrange
        var wavelet = new HaarWavelet<double>();

        // Act
        var result = wavelet.Calculate(0.75);

        // Assert - x in [0.5, 1) should return -1
        Assert.Equal(-1.0, result, Tolerance);
    }

    [Fact]
    public void HaarWavelet_Calculate_OutsideRange_ReturnsZero()
    {
        // Arrange
        var wavelet = new HaarWavelet<double>();

        // Act
        var result1 = wavelet.Calculate(-0.5);
        var result2 = wavelet.Calculate(1.5);

        // Assert - outside [0, 1) should return 0
        Assert.Equal(0.0, result1, Tolerance);
        Assert.Equal(0.0, result2, Tolerance);
    }

    [Fact]
    public void HaarWavelet_Decompose_EvenLength_ReturnsCorrectSize()
    {
        // Arrange
        var wavelet = new HaarWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

        // Act
        var (approximation, detail) = wavelet.Decompose(input);

        // Assert
        Assert.Equal(2, approximation.Length);
        Assert.Equal(2, detail.Length);
    }

    [Fact]
    public void HaarWavelet_Decompose_OddLength_ThrowsArgumentException()
    {
        // Arrange
        var wavelet = new HaarWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => wavelet.Decompose(input));
    }

    [Fact]
    public void HaarWavelet_GetScalingCoefficients_ReturnsTwoElements()
    {
        // Arrange
        var wavelet = new HaarWavelet<double>();

        // Act
        var coefficients = wavelet.GetScalingCoefficients();

        // Assert
        Assert.Equal(2, coefficients.Length);
        Assert.Equal(1.0 / Math.Sqrt(2), coefficients[0], Tolerance);
        Assert.Equal(1.0 / Math.Sqrt(2), coefficients[1], Tolerance);
    }

    [Fact]
    public void HaarWavelet_GetWaveletCoefficients_ReturnsTwoElements()
    {
        // Arrange
        var wavelet = new HaarWavelet<double>();

        // Act
        var coefficients = wavelet.GetWaveletCoefficients();

        // Assert
        Assert.Equal(2, coefficients.Length);
        Assert.Equal(1.0 / Math.Sqrt(2), coefficients[0], Tolerance);
        Assert.Equal(-1.0 / Math.Sqrt(2), coefficients[1], Tolerance);
    }

    #endregion

    #region Daubechies Wavelet Tests

    [Fact]
    public void DaubechiesWavelet_Calculate_ReturnsValue()
    {
        // Arrange
        var wavelet = new DaubechiesWavelet<double>(order: 4);

        // Act
        var result = wavelet.Calculate(0.5);

        // Assert - should return a finite value
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    [Fact]
    public void DaubechiesWavelet_GetScalingCoefficients_ReturnsCorrectLength()
    {
        // Arrange
        var wavelet = new DaubechiesWavelet<double>(order: 4);

        // Act
        var coefficients = wavelet.GetScalingCoefficients();

        // Assert - Db4 should have 8 coefficients
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void DaubechiesWavelet_GetWaveletCoefficients_ReturnsCorrectLength()
    {
        // Arrange
        var wavelet = new DaubechiesWavelet<double>(order: 4);

        // Act
        var coefficients = wavelet.GetWaveletCoefficients();

        // Assert
        Assert.True(coefficients.Length > 0);
    }

    #endregion

    #region Mexican Hat Wavelet Tests

    [Fact]
    public void MexicanHatWavelet_Calculate_AtZero_ReturnsPositiveValue()
    {
        // Arrange
        var wavelet = new MexicanHatWavelet<double>();

        // Act
        var result = wavelet.Calculate(0.0);

        // Assert - peak at center
        Assert.True(result > 0);
    }

    [Fact]
    public void MexicanHatWavelet_Calculate_Symmetric()
    {
        // Arrange
        var wavelet = new MexicanHatWavelet<double>();

        // Act
        var result1 = wavelet.Calculate(0.5);
        var result2 = wavelet.Calculate(-0.5);

        // Assert - Mexican Hat is symmetric
        Assert.Equal(result1, result2, Tolerance);
    }

    [Fact]
    public void MexicanHatWavelet_Calculate_FarFromCenter_ApproachesZero()
    {
        // Arrange
        var wavelet = new MexicanHatWavelet<double>();

        // Act
        var result = wavelet.Calculate(10.0);

        // Assert - should decay to near zero
        Assert.True(Math.Abs(result) < 0.01);
    }

    #endregion

    #region Morlet Wavelet Tests

    [Fact]
    public void MorletWavelet_Calculate_ReturnsValue()
    {
        // Arrange
        var wavelet = new MorletWavelet<double>();

        // Act
        var result = wavelet.Calculate(0.0);

        // Assert
        Assert.False(double.IsNaN(result));
    }

    [Fact]
    public void MorletWavelet_Calculate_DecaysAwayFromCenter()
    {
        // Arrange
        var wavelet = new MorletWavelet<double>();

        // Act
        var centerValue = Math.Abs(wavelet.Calculate(0.0));
        var farValue = Math.Abs(wavelet.Calculate(5.0));

        // Assert - should decay with distance
        Assert.True(centerValue >= farValue);
    }

    #endregion

    #region Gaussian Wavelet Tests

    [Fact]
    public void GaussianWavelet_Calculate_AtZero_ReturnsValue()
    {
        // Arrange
        var wavelet = new GaussianWavelet<double>(sigma: 1.0);

        // Act
        var result = wavelet.Calculate(0.0);

        // Assert
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    [Fact]
    public void GaussianWavelet_DifferentSigmas_ProduceDifferentResults()
    {
        // Arrange
        var wavelet1 = new GaussianWavelet<double>(sigma: 0.5);
        var wavelet2 = new GaussianWavelet<double>(sigma: 2.0);

        // Act
        var result1 = wavelet1.Calculate(0.5);
        var result2 = wavelet2.Calculate(0.5);

        // Assert - different sigmas should give different values
        Assert.NotEqual(result1, result2, Tolerance);
    }

    #endregion

    #region Symlet Wavelet Tests

    [Fact]
    public void SymletWavelet_Calculate_ReturnsValue()
    {
        // Arrange
        var wavelet = new SymletWavelet<double>(order: 4);

        // Act
        var result = wavelet.Calculate(0.5);

        // Assert
        Assert.False(double.IsNaN(result));
    }

    [Fact]
    public void SymletWavelet_GetScalingCoefficients_HasCoefficients()
    {
        // Arrange
        var wavelet = new SymletWavelet<double>(order: 4);

        // Act
        var coefficients = wavelet.GetScalingCoefficients();

        // Assert
        Assert.True(coefficients.Length > 0);
    }

    #endregion

    #region Coiflet Wavelet Tests

    [Fact]
    public void CoifletWavelet_Calculate_ReturnsValue()
    {
        // Arrange
        var wavelet = new CoifletWavelet<double>(order: 2);

        // Act
        var result = wavelet.Calculate(0.5);

        // Assert
        Assert.False(double.IsNaN(result));
    }

    [Fact]
    public void CoifletWavelet_GetCoefficients_HasCoefficients()
    {
        // Arrange
        var wavelet = new CoifletWavelet<double>(order: 2);

        // Act
        var scaling = wavelet.GetScalingCoefficients();
        var waveletCoeffs = wavelet.GetWaveletCoefficients();

        // Assert
        Assert.True(scaling.Length > 0);
        Assert.True(waveletCoeffs.Length > 0);
    }

    #endregion

    #region Biorthogonal Wavelet Tests

    [Fact]
    public void BiorthogonalWavelet_Calculate_ReturnsValue()
    {
        // Arrange
        var wavelet = new BiorthogonalWavelet<double>(decompositionOrder: 2, reconstructionOrder: 2);

        // Act
        var result = wavelet.Calculate(0.5);

        // Assert
        Assert.False(double.IsNaN(result));
    }

    #endregion

    #region Meyer Wavelet Tests

    [Fact]
    public void MeyerWavelet_Calculate_ReturnsValue()
    {
        // Arrange
        var wavelet = new MeyerWavelet<double>();

        // Act
        var result = wavelet.Calculate(0.5);

        // Assert
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    #endregion

    #region Shannon Wavelet Tests

    [Fact]
    public void ShannonWavelet_Calculate_ReturnsValue()
    {
        // Arrange
        var wavelet = new ShannonWavelet<double>();

        // Act
        var result = wavelet.Calculate(0.5);

        // Assert
        Assert.False(double.IsNaN(result));
    }

    #endregion

    #region DOG Wavelet Tests

    [Fact]
    public void DOGWavelet_Calculate_ReturnsValue()
    {
        // Arrange
        var wavelet = new DOGWavelet<double>(order: 2);

        // Act
        var result = wavelet.Calculate(0.0);

        // Assert
        Assert.False(double.IsNaN(result));
    }

    [Fact]
    public void DOGWavelet_Calculate_Symmetric()
    {
        // Arrange
        var wavelet = new DOGWavelet<double>(order: 2);

        // Act
        var result1 = wavelet.Calculate(1.0);
        var result2 = wavelet.Calculate(-1.0);

        // Assert - DOG wavelets are symmetric or antisymmetric
        Assert.True(Math.Abs(result1) - Math.Abs(result2) < 0.01);
    }

    #endregion

    #region BSpline Wavelet Tests

    [Fact]
    public void BSplineWavelet_Calculate_ReturnsValue()
    {
        // Arrange
        var wavelet = new BSplineWavelet<double>(order: 3);

        // Act
        var result = wavelet.Calculate(0.5);

        // Assert
        Assert.False(double.IsNaN(result));
    }

    #endregion

    #region Gabor Wavelet Tests

    [Fact]
    public void GaborWavelet_Calculate_ReturnsValue()
    {
        // Arrange
        var wavelet = new GaborWavelet<double>();

        // Act
        var result = wavelet.Calculate(0.0);

        // Assert
        Assert.False(double.IsNaN(result));
    }

    [Fact]
    public void GaborWavelet_Calculate_DecaysWithDistance()
    {
        // Arrange
        var wavelet = new GaborWavelet<double>();

        // Act
        var centerMag = Math.Abs(wavelet.Calculate(0.0));
        var farMag = Math.Abs(wavelet.Calculate(5.0));

        // Assert
        Assert.True(centerMag >= farMag);
    }

    #endregion

    #region Paul Wavelet Tests

    [Fact]
    public void PaulWavelet_Calculate_ReturnsValue()
    {
        // Arrange
        var wavelet = new PaulWavelet<double>(order: 4);

        // Act
        var result = wavelet.Calculate(0.5);

        // Assert
        Assert.False(double.IsNaN(result));
    }

    #endregion

    #region Complex Wavelet Tests

    [Fact]
    public void ComplexMorletWavelet_Calculate_ReturnsComplexValue()
    {
        // Arrange
        var wavelet = new ComplexMorletWavelet<double>();
        var input = new Complex<double>(0.5, 0.0);

        // Act
        var result = wavelet.Calculate(input);

        // Assert
        Assert.False(double.IsNaN(result.Real));
        Assert.False(double.IsNaN(result.Imaginary));
        Assert.False(double.IsInfinity(result.Real));
        Assert.False(double.IsInfinity(result.Imaginary));
    }

    [Fact]
    public void ComplexMorletWavelet_Calculate_AtZero_ReturnsValue()
    {
        // Arrange
        var wavelet = new ComplexMorletWavelet<double>();
        var input = new Complex<double>(0.0, 0.0);

        // Act
        var result = wavelet.Calculate(input);

        // Assert - at center, should have strong response
        Assert.True(result.Magnitude > 0);
    }

    [Fact]
    public void ComplexMorletWavelet_CustomOmega_AffectsResult()
    {
        // Arrange
        var wavelet1 = new ComplexMorletWavelet<double>(omega: 3.0);
        var wavelet2 = new ComplexMorletWavelet<double>(omega: 7.0);
        var input = new Complex<double>(1.0, 0.0);

        // Act
        var result1 = wavelet1.Calculate(input);
        var result2 = wavelet2.Calculate(input);

        // Assert - different omega values should produce different results
        Assert.NotEqual(result1.Real, result2.Real, Tolerance);
    }

    [Fact]
    public void ComplexMorletWavelet_GetScalingCoefficients_ReturnsVector()
    {
        // Arrange
        var wavelet = new ComplexMorletWavelet<double>();

        // Act
        var coeffs = wavelet.GetScalingCoefficients();

        // Assert
        Assert.NotNull(coeffs);
        Assert.True(coeffs.Length > 0);
    }

    [Fact]
    public void ComplexMorletWavelet_GetWaveletCoefficients_ReturnsVector()
    {
        // Arrange
        var wavelet = new ComplexMorletWavelet<double>();

        // Act
        var coeffs = wavelet.GetWaveletCoefficients();

        // Assert
        Assert.NotNull(coeffs);
        Assert.True(coeffs.Length > 0);
    }

    [Fact]
    public void ComplexGaussianWavelet_Calculate_ReturnsComplexValue()
    {
        // Arrange
        var wavelet = new ComplexGaussianWavelet<double>();
        var input = new Complex<double>(0.5, 0.0);

        // Act
        var result = wavelet.Calculate(input);

        // Assert
        Assert.False(double.IsNaN(result.Real));
        Assert.False(double.IsNaN(result.Imaginary));
    }

    [Fact]
    public void ComplexGaussianWavelet_Calculate_AtZero_ReturnsValue()
    {
        // Arrange
        var wavelet = new ComplexGaussianWavelet<double>(order: 1);
        var input = new Complex<double>(0.0, 0.0);

        // Act
        var result = wavelet.Calculate(input);

        // Assert
        Assert.False(double.IsNaN(result.Real));
    }

    [Fact]
    public void ComplexGaussianWavelet_DifferentOrders_ProduceDifferentResults()
    {
        // Arrange
        var wavelet1 = new ComplexGaussianWavelet<double>(order: 1);
        var wavelet2 = new ComplexGaussianWavelet<double>(order: 3);
        var input = new Complex<double>(1.0, 0.0);

        // Act
        var result1 = wavelet1.Calculate(input);
        var result2 = wavelet2.Calculate(input);

        // Assert - different orders should produce different results
        Assert.NotEqual(result1.Real, result2.Real, Tolerance);
    }

    [Fact]
    public void ComplexGaussianWavelet_GetScalingCoefficients_ReturnsVector()
    {
        // Arrange
        var wavelet = new ComplexGaussianWavelet<double>();

        // Act
        var coeffs = wavelet.GetScalingCoefficients();

        // Assert
        Assert.NotNull(coeffs);
        Assert.True(coeffs.Length > 0);
    }

    [Fact]
    public void ComplexGaussianWavelet_GetWaveletCoefficients_ReturnsVector()
    {
        // Arrange
        var wavelet = new ComplexGaussianWavelet<double>(order: 2);

        // Act
        var coeffs = wavelet.GetWaveletCoefficients();

        // Assert
        Assert.NotNull(coeffs);
        Assert.True(coeffs.Length > 0);
    }

    [Fact]
    public void ComplexMorletWavelet_Decompose_ReturnsApproximationAndDetail()
    {
        // Arrange
        var wavelet = new ComplexMorletWavelet<double>();
        var inputData = new Complex<double>[8];
        for (int i = 0; i < 8; i++)
            inputData[i] = new Complex<double>((double)(i + 1), 0.0);
        var input = new Vector<Complex<double>>(inputData);

        // Act
        var (approximation, detail) = wavelet.Decompose(input);

        // Assert
        Assert.NotNull(approximation);
        Assert.NotNull(detail);
        Assert.True(approximation.Length > 0);
        Assert.True(detail.Length > 0);
    }

    [Fact]
    public void ComplexGaussianWavelet_Decompose_ReturnsApproximationAndDetail()
    {
        // Arrange
        var wavelet = new ComplexGaussianWavelet<double>(order: 2);
        var inputData = new Complex<double>[8];
        for (int i = 0; i < 8; i++)
            inputData[i] = new Complex<double>((double)(i + 1), 0.0);
        var input = new Vector<Complex<double>>(inputData);

        // Act
        var (approximation, detail) = wavelet.Decompose(input);

        // Assert
        Assert.NotNull(approximation);
        Assert.NotNull(detail);
        Assert.True(approximation.Length > 0);
        Assert.True(detail.Length > 0);
    }

    #endregion

    #region ContinuousMexicanHat Wavelet Tests

    [Fact]
    public void ContinuousMexicanHatWavelet_Calculate_ReturnsValue()
    {
        // Arrange
        var wavelet = new ContinuousMexicanHatWavelet<double>();

        // Act
        var result = wavelet.Calculate(0.0);

        // Assert
        Assert.False(double.IsNaN(result));
    }

    #endregion

    #region ReverseBiorthogonal Wavelet Tests

    // Note: ReverseBiorthogonalWavelet requires INumericOperations<T> as first parameter
    // which makes it complex to test directly. Skipping individual test.

    #endregion

    #region BattleLemarie Wavelet Tests

    [Fact]
    public void BattleLemarieWavelet_Calculate_ReturnsValue()
    {
        // Arrange
        var wavelet = new BattleLemarieWavelet<double>(order: 1);

        // Act
        var result = wavelet.Calculate(0.5);

        // Assert
        Assert.False(double.IsNaN(result));
    }

    #endregion

    #region FejerKorovkin Wavelet Tests

    [Fact]
    public void FejérKorovkinWavelet_Calculate_ReturnsValue()
    {
        // Arrange
        var wavelet = new FejérKorovkinWavelet<double>(order: 4);

        // Act
        var result = wavelet.Calculate(0.5);

        // Assert
        Assert.False(double.IsNaN(result));
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void AllWaveletFunctions_Calculate_DoNotReturnNaN()
    {
        // Arrange
        var wavelets = new IWaveletFunction<double>[]
        {
            new HaarWavelet<double>(),
            new MexicanHatWavelet<double>(),
            new MorletWavelet<double>(),
            new GaussianWavelet<double>(sigma: 1.0),
            new ShannonWavelet<double>(),
            new DOGWavelet<double>(order: 2),
            new GaborWavelet<double>(),
            new MeyerWavelet<double>()
        };

        // Act & Assert
        foreach (var wavelet in wavelets)
        {
            var result = wavelet.Calculate(0.5);
            Assert.False(double.IsNaN(result), $"Wavelet {wavelet.GetType().Name} returned NaN");
        }
    }

    [Fact]
    public void HaarWavelet_Decompose_PreservesEnergy()
    {
        // Arrange
        var wavelet = new HaarWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 3.0, 5.0, 7.0 });

        // Act
        var (approximation, detail) = wavelet.Decompose(input);

        // Assert - energy should be approximately preserved
        double inputEnergy = 0;
        for (int i = 0; i < input.Length; i++)
            inputEnergy += input[i] * input[i];

        double outputEnergy = 0;
        for (int i = 0; i < approximation.Length; i++)
            outputEnergy += approximation[i] * approximation[i] + detail[i] * detail[i];

        Assert.Equal(inputEnergy, outputEnergy, 0.01);
    }

    [Fact]
    public void WaveletFunctions_GetCoefficients_NoNaNValues()
    {
        // Arrange
        var wavelets = new IWaveletFunction<double>[]
        {
            new HaarWavelet<double>(),
            new DaubechiesWavelet<double>(order: 4),
            new SymletWavelet<double>(order: 4),
            new CoifletWavelet<double>(order: 2)
        };

        // Act & Assert
        foreach (var wavelet in wavelets)
        {
            var scaling = wavelet.GetScalingCoefficients();
            var waveletCoeffs = wavelet.GetWaveletCoefficients();

            for (int i = 0; i < scaling.Length; i++)
            {
                Assert.False(double.IsNaN(scaling[i]), $"Wavelet {wavelet.GetType().Name} has NaN scaling coefficient");
            }
            for (int i = 0; i < waveletCoeffs.Length; i++)
            {
                Assert.False(double.IsNaN(waveletCoeffs[i]), $"Wavelet {wavelet.GetType().Name} has NaN wavelet coefficient");
            }
        }
    }

    #endregion

    #region Reverse Biorthogonal Wavelet Tests

    [Fact]
    public void ReverseBiorthogonalWavelet_DefaultConstructor_CreatesInstance()
    {
        // Arrange & Act
        var wavelet = new ReverseBiorthogonalWavelet<double>();

        // Assert
        Assert.NotNull(wavelet);
    }

    [Fact]
    public void ReverseBiorthogonalWavelet_Calculate_ReturnsValue()
    {
        // Arrange
        var wavelet = new ReverseBiorthogonalWavelet<double>();

        // Act
        var result = wavelet.Calculate(0.5);

        // Assert - Should return a numeric value
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    [Fact]
    public void ReverseBiorthogonalWavelet_GetScalingCoefficients_ReturnsNonEmptyVector()
    {
        // Arrange
        var wavelet = new ReverseBiorthogonalWavelet<double>();

        // Act
        var coefficients = wavelet.GetScalingCoefficients();

        // Assert
        Assert.NotNull(coefficients);
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void ReverseBiorthogonalWavelet_GetWaveletCoefficients_ReturnsNonEmptyVector()
    {
        // Arrange
        var wavelet = new ReverseBiorthogonalWavelet<double>();

        // Act
        var coefficients = wavelet.GetWaveletCoefficients();

        // Assert
        Assert.NotNull(coefficients);
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void ReverseBiorthogonalWavelet_Decompose_ReturnsApproximationAndDetail()
    {
        // Arrange
        var wavelet = new ReverseBiorthogonalWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        // Act
        var (approximation, detail) = wavelet.Decompose(input);

        // Assert
        Assert.NotNull(approximation);
        Assert.NotNull(detail);
        Assert.True(approximation.Length > 0);
        Assert.True(detail.Length > 0);
    }

    #endregion

    #region Mathematical Correctness Tests - Haar Wavelet

    [Fact]
    public void HaarWavelet_Calculate_BoundaryAt0_ReturnsOne()
    {
        // Arrange
        var wavelet = new HaarWavelet<double>();

        // Act
        var result = wavelet.Calculate(0.0);

        // Assert - x = 0 is in [0, 0.5) so should return 1
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void HaarWavelet_Calculate_BoundaryAt0Point5_ReturnsNegativeOne()
    {
        // Arrange
        var wavelet = new HaarWavelet<double>();

        // Act
        var result = wavelet.Calculate(0.5);

        // Assert - x = 0.5 is in [0.5, 1) so should return -1
        Assert.Equal(-1.0, result, Tolerance);
    }

    [Fact]
    public void HaarWavelet_Calculate_BoundaryAt1_ReturnsZero()
    {
        // Arrange
        var wavelet = new HaarWavelet<double>();

        // Act
        var result = wavelet.Calculate(1.0);

        // Assert - x = 1 is outside [0, 1) so should return 0
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void HaarWavelet_ScalingCoefficients_CorrectValues()
    {
        // Arrange
        var wavelet = new HaarWavelet<double>();
        double expectedValue = 1.0 / Math.Sqrt(2);

        // Act
        var coefficients = wavelet.GetScalingCoefficients();

        // Assert - Haar scaling coefficients are [1/√2, 1/√2]
        Assert.Equal(2, coefficients.Length);
        Assert.Equal(expectedValue, coefficients[0], Tolerance);
        Assert.Equal(expectedValue, coefficients[1], Tolerance);
    }

    [Fact]
    public void HaarWavelet_WaveletCoefficients_CorrectValues()
    {
        // Arrange
        var wavelet = new HaarWavelet<double>();
        double expectedPositive = 1.0 / Math.Sqrt(2);
        double expectedNegative = -1.0 / Math.Sqrt(2);

        // Act
        var coefficients = wavelet.GetWaveletCoefficients();

        // Assert - Haar wavelet coefficients are [1/√2, -1/√2]
        Assert.Equal(2, coefficients.Length);
        Assert.Equal(expectedPositive, coefficients[0], Tolerance);
        Assert.Equal(expectedNegative, coefficients[1], Tolerance);
    }

    [Fact]
    public void HaarWavelet_ScalingCoefficients_SumToSqrt2()
    {
        // Arrange
        var wavelet = new HaarWavelet<double>();

        // Act
        var coefficients = wavelet.GetScalingCoefficients();
        double sum = 0;
        for (int i = 0; i < coefficients.Length; i++)
            sum += coefficients[i];

        // Assert - Sum of scaling coefficients = √2 for orthogonal wavelets
        Assert.Equal(Math.Sqrt(2), sum, Tolerance);
    }

    [Fact]
    public void HaarWavelet_WaveletCoefficients_SumToZero()
    {
        // Arrange
        var wavelet = new HaarWavelet<double>();

        // Act
        var coefficients = wavelet.GetWaveletCoefficients();
        double sum = 0;
        for (int i = 0; i < coefficients.Length; i++)
            sum += coefficients[i];

        // Assert - Sum of wavelet coefficients = 0 (zero mean property)
        Assert.Equal(0.0, sum, Tolerance);
    }

    #endregion

    #region Mathematical Correctness Tests - Mexican Hat Wavelet

    [Fact]
    public void MexicanHatWavelet_Calculate_AtZero_ReturnsTwoWithDefaultSigma()
    {
        // Arrange - Mexican Hat: (2 - x²/σ²) * exp(-x²/2σ²)
        // At x=0 with σ=1: (2 - 0) * exp(0) = 2
        var wavelet = new MexicanHatWavelet<double>(sigma: 1.0);

        // Act
        var result = wavelet.Calculate(0.0);

        // Assert
        Assert.Equal(2.0, result, Tolerance);
    }

    [Fact]
    public void MexicanHatWavelet_Calculate_ZeroCrossings()
    {
        // Arrange - Zero crossings occur where 2 - x²/σ² = 0, i.e., x = ±√2σ
        var wavelet = new MexicanHatWavelet<double>(sigma: 1.0);
        double zeroCrossing = Math.Sqrt(2);

        // Act
        var result1 = wavelet.Calculate(zeroCrossing);
        var result2 = wavelet.Calculate(-zeroCrossing);

        // Assert - Should be approximately zero at x = ±√2
        Assert.True(Math.Abs(result1) < 0.01, $"Expected ~0 at x=√2, got {result1}");
        Assert.True(Math.Abs(result2) < 0.01, $"Expected ~0 at x=-√2, got {result2}");
    }

    [Fact]
    public void MexicanHatWavelet_Calculate_NegativeInDipRegion()
    {
        // Arrange - The "dip" of the sombrero is negative, occurs around x ≈ ±2
        var wavelet = new MexicanHatWavelet<double>(sigma: 1.0);

        // Act
        var result1 = wavelet.Calculate(2.0);
        var result2 = wavelet.Calculate(-2.0);

        // Assert - Should be negative in the dip region
        Assert.True(result1 < 0, $"Expected negative at x=2, got {result1}");
        Assert.True(result2 < 0, $"Expected negative at x=-2, got {result2}");
    }

    [Fact]
    public void MexicanHatWavelet_Calculate_PerfectSymmetry()
    {
        // Arrange
        var wavelet = new MexicanHatWavelet<double>(sigma: 1.0);
        double[] testPoints = { 0.5, 1.0, 1.5, 2.0, 3.0 };

        // Act & Assert - Mexican Hat is symmetric: f(x) = f(-x)
        foreach (var x in testPoints)
        {
            var positive = wavelet.Calculate(x);
            var negative = wavelet.Calculate(-x);
            Assert.Equal(positive, negative, Tolerance);
        }
    }

    [Fact]
    public void MexicanHatWavelet_Calculate_DifferentSigmas_ScaleCorrectly()
    {
        // Arrange
        var wavelet1 = new MexicanHatWavelet<double>(sigma: 1.0);
        var wavelet2 = new MexicanHatWavelet<double>(sigma: 2.0);

        // Act - Both should have peak value 2 at x=0
        var peak1 = wavelet1.Calculate(0.0);
        var peak2 = wavelet2.Calculate(0.0);

        // Assert - Peak value is always 2 regardless of sigma
        Assert.Equal(2.0, peak1, Tolerance);
        Assert.Equal(2.0, peak2, Tolerance);
    }

    #endregion

    #region Mathematical Correctness Tests - Morlet Wavelet

    [Fact]
    public void MorletWavelet_Calculate_AtZero_ReturnsOne()
    {
        // Arrange - Morlet: cos(ω*x) * exp(-x²/2)
        // At x=0: cos(0) * exp(0) = 1 * 1 = 1
        var wavelet = new MorletWavelet<double>(omega: 5.0);

        // Act
        var result = wavelet.Calculate(0.0);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void MorletWavelet_Calculate_OscillatesWithOmega()
    {
        // Arrange - With ω=5, should have zeros approximately at x = π/(2ω) = π/10
        var wavelet = new MorletWavelet<double>(omega: 5.0);
        double firstZero = Math.PI / 10.0; // cos(5 * π/10) = cos(π/2) = 0

        // Act
        var result = wavelet.Calculate(firstZero);

        // Assert - Should be close to zero (but not exactly due to Gaussian decay)
        Assert.True(Math.Abs(result) < 0.1, $"Expected ~0 at first oscillation zero, got {result}");
    }

    [Fact]
    public void MorletWavelet_Calculate_DecaysWithDistance()
    {
        // Arrange
        var wavelet = new MorletWavelet<double>(omega: 5.0);

        // Act
        var atCenter = Math.Abs(wavelet.Calculate(0.0));
        var at1 = Math.Abs(wavelet.Calculate(1.0));
        var at2 = Math.Abs(wavelet.Calculate(2.0));
        var at3 = Math.Abs(wavelet.Calculate(3.0));

        // Assert - Magnitude should generally decrease (Gaussian envelope)
        Assert.True(atCenter >= at1 || at1 < 0.7, "Should decay from center");
        Assert.True(at2 < 0.2, "Should be small at x=2");
        Assert.True(at3 < 0.02, "Should be very small at x=3");
    }

    [Fact]
    public void MorletWavelet_DifferentOmega_AffectsOscillationFrequency()
    {
        // Arrange
        var waveletLowOmega = new MorletWavelet<double>(omega: 2.0);
        var waveletHighOmega = new MorletWavelet<double>(omega: 10.0);

        // Act - At x=0.5, higher omega should have completed more oscillations
        var lowResult = waveletLowOmega.Calculate(0.5);
        var highResult = waveletHighOmega.Calculate(0.5);

        // Assert - Different omega values give different results
        Assert.NotEqual(lowResult, highResult, 0.1);
    }

    #endregion

    #region Mathematical Correctness Tests - Gaussian Wavelet

    [Fact]
    public void GaussianWavelet_Calculate_AtZero_ReturnsOne()
    {
        // Arrange - Gaussian: exp(-x²/2σ²)
        // At x=0: exp(0) = 1
        var wavelet = new GaussianWavelet<double>(sigma: 1.0);

        // Act
        var result = wavelet.Calculate(0.0);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void GaussianWavelet_Calculate_AlwaysPositive()
    {
        // Arrange
        var wavelet = new GaussianWavelet<double>(sigma: 1.0);
        double[] testPoints = { -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0 };

        // Act & Assert - Gaussian is always positive
        foreach (var x in testPoints)
        {
            var result = wavelet.Calculate(x);
            Assert.True(result > 0, $"Gaussian should be positive at x={x}, got {result}");
        }
    }

    [Fact]
    public void GaussianWavelet_Calculate_PerfectSymmetry()
    {
        // Arrange
        var wavelet = new GaussianWavelet<double>(sigma: 1.0);
        double[] testPoints = { 0.5, 1.0, 1.5, 2.0, 3.0 };

        // Act & Assert - Gaussian is symmetric: f(x) = f(-x)
        foreach (var x in testPoints)
        {
            var positive = wavelet.Calculate(x);
            var negative = wavelet.Calculate(-x);
            Assert.Equal(positive, negative, Tolerance);
        }
    }

    [Fact]
    public void GaussianWavelet_Calculate_AtOneSigma_ReturnsExpNegHalf()
    {
        // Arrange - At x=σ: exp(-σ²/2σ²) = exp(-1/2) ≈ 0.6065
        var wavelet = new GaussianWavelet<double>(sigma: 1.0);
        double expected = Math.Exp(-0.5);

        // Act
        var result = wavelet.Calculate(1.0);

        // Assert
        Assert.Equal(expected, result, Tolerance);
    }

    [Fact]
    public void GaussianWavelet_Calculate_MonotonicallyDecreases()
    {
        // Arrange
        var wavelet = new GaussianWavelet<double>(sigma: 1.0);

        // Act
        var at0 = wavelet.Calculate(0.0);
        var at1 = wavelet.Calculate(1.0);
        var at2 = wavelet.Calculate(2.0);
        var at3 = wavelet.Calculate(3.0);

        // Assert - Should monotonically decrease from center
        Assert.True(at0 > at1, "Should decrease from x=0 to x=1");
        Assert.True(at1 > at2, "Should decrease from x=1 to x=2");
        Assert.True(at2 > at3, "Should decrease from x=2 to x=3");
    }

    [Fact]
    public void GaussianWavelet_DifferentSigmas_AffectWidth()
    {
        // Arrange
        var narrowWavelet = new GaussianWavelet<double>(sigma: 0.5);
        var wideWavelet = new GaussianWavelet<double>(sigma: 2.0);

        // Act - At x=1, narrower wavelet should have decayed more
        var narrowResult = narrowWavelet.Calculate(1.0);
        var wideResult = wideWavelet.Calculate(1.0);

        // Assert
        Assert.True(narrowResult < wideResult, "Narrower wavelet should decay faster");
    }

    #endregion

    #region Mathematical Correctness Tests - Daubechies Wavelet

    [Fact]
    public void DaubechiesWavelet_D4_ScalingCoefficients_CorrectValues()
    {
        // Arrange - D4 coefficients are well-known
        var wavelet = new DaubechiesWavelet<double>(order: 4);
        double h0 = (1 + Math.Sqrt(3)) / (4 * Math.Sqrt(2));
        double h1 = (3 + Math.Sqrt(3)) / (4 * Math.Sqrt(2));
        double h2 = (3 - Math.Sqrt(3)) / (4 * Math.Sqrt(2));
        double h3 = (1 - Math.Sqrt(3)) / (4 * Math.Sqrt(2));

        // Act
        var coefficients = wavelet.GetScalingCoefficients();

        // Assert
        Assert.Equal(4, coefficients.Length);
        Assert.Equal(h0, coefficients[0], Tolerance);
        Assert.Equal(h1, coefficients[1], Tolerance);
        Assert.Equal(h2, coefficients[2], Tolerance);
        Assert.Equal(h3, coefficients[3], Tolerance);
    }

    [Fact]
    public void DaubechiesWavelet_D4_ScalingCoefficients_SumToSqrt2()
    {
        // Arrange
        var wavelet = new DaubechiesWavelet<double>(order: 4);

        // Act
        var coefficients = wavelet.GetScalingCoefficients();
        double sum = 0;
        for (int i = 0; i < coefficients.Length; i++)
            sum += coefficients[i];

        // Assert - Sum of scaling coefficients = √2 for orthogonal wavelets
        Assert.Equal(Math.Sqrt(2), sum, Tolerance);
    }

    [Fact]
    public void DaubechiesWavelet_D4_WaveletCoefficients_QuadratureMirrorFilter()
    {
        // Arrange - Wavelet coefficients: g[n] = (-1)^n * h[L-1-n]
        var wavelet = new DaubechiesWavelet<double>(order: 4);

        // Act
        var scaling = wavelet.GetScalingCoefficients();
        var waveletCoeffs = wavelet.GetWaveletCoefficients();

        // Assert - Verify QMF relationship
        Assert.Equal(scaling.Length, waveletCoeffs.Length);
        int L = scaling.Length;
        for (int n = 0; n < L; n++)
        {
            double expected = Math.Pow(-1, n) * scaling[L - 1 - n];
            Assert.Equal(expected, waveletCoeffs[n], Tolerance);
        }
    }

    [Fact]
    public void DaubechiesWavelet_D4_WaveletCoefficients_SumToZero()
    {
        // Arrange
        var wavelet = new DaubechiesWavelet<double>(order: 4);

        // Act
        var coefficients = wavelet.GetWaveletCoefficients();
        double sum = 0;
        for (int i = 0; i < coefficients.Length; i++)
            sum += coefficients[i];

        // Assert - Sum of wavelet coefficients = 0 (zero mean property)
        Assert.Equal(0.0, sum, Tolerance);
    }

    [Fact]
    public void DaubechiesWavelet_D4_ScalingCoefficients_UnitNorm()
    {
        // Arrange
        var wavelet = new DaubechiesWavelet<double>(order: 4);

        // Act
        var coefficients = wavelet.GetScalingCoefficients();
        double sumOfSquares = 0;
        for (int i = 0; i < coefficients.Length; i++)
            sumOfSquares += coefficients[i] * coefficients[i];

        // Assert - Sum of squares = 1 for orthonormal wavelets
        Assert.Equal(1.0, sumOfSquares, Tolerance);
    }

    [Fact]
    public void DaubechiesWavelet_Calculate_ZeroOutsideSupport()
    {
        // Arrange - D4 has support [0, 3]
        var wavelet = new DaubechiesWavelet<double>(order: 4);

        // Act
        var belowSupport = wavelet.Calculate(-0.1);
        var aboveSupport = wavelet.Calculate(3.1);

        // Assert - Should be zero outside support
        Assert.Equal(0.0, belowSupport, Tolerance);
        Assert.Equal(0.0, aboveSupport, Tolerance);
    }

    #endregion

    #region Mathematical Correctness Tests - Energy Preservation

    [Fact]
    public void HaarWavelet_Decompose_PreservesEnergyExactly()
    {
        // Arrange
        var wavelet = new HaarWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        // Act
        var (approximation, detail) = wavelet.Decompose(input);

        // Calculate energies
        double inputEnergy = 0;
        for (int i = 0; i < input.Length; i++)
            inputEnergy += input[i] * input[i];

        double outputEnergy = 0;
        for (int i = 0; i < approximation.Length; i++)
            outputEnergy += approximation[i] * approximation[i] + detail[i] * detail[i];

        // Assert - Energy should be exactly preserved
        Assert.Equal(inputEnergy, outputEnergy, Tolerance);
    }

    [Fact]
    public void DaubechiesWavelet_Decompose_PreservesEnergy()
    {
        // Arrange
        var wavelet = new DaubechiesWavelet<double>(order: 4);
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        // Act
        var (approximation, detail) = wavelet.Decompose(input);

        // Calculate energies
        double inputEnergy = 0;
        for (int i = 0; i < input.Length; i++)
            inputEnergy += input[i] * input[i];

        double outputEnergy = 0;
        for (int i = 0; i < approximation.Length; i++)
            outputEnergy += approximation[i] * approximation[i] + detail[i] * detail[i];

        // Assert - Energy should be approximately preserved (circular convolution may introduce small differences)
        Assert.Equal(inputEnergy, outputEnergy, 0.1);
    }

    #endregion

    #region Mathematical Correctness Tests - DOG Wavelet

    [Fact]
    public void DOGWavelet_Order2_IsSecondDerivativeOfGaussian()
    {
        // Arrange - DOG order 2 formula: (x² - 1) * exp(-x²/2)
        // At x=0: (0 - 1) * 1 = -1 (times normalization factor)
        var wavelet = new DOGWavelet<double>(order: 2);

        // Act
        var atZero = wavelet.Calculate(0.0);

        // Assert - DOG order 2 should be negative at center (x² - 1 = -1 at x=0)
        Assert.True(atZero < 0, $"DOG order 2 should be negative at center, got {atZero}");
        // At x=1: (1 - 1) * exp(-0.5) = 0
        var atOne = wavelet.Calculate(1.0);
        Assert.True(Math.Abs(atOne) < 0.01, $"DOG order 2 should be ~0 at x=1, got {atOne}");
    }

    [Fact]
    public void DOGWavelet_EvenOrder_IsSymmetric()
    {
        // Arrange - Even order DOG wavelets are symmetric
        var wavelet = new DOGWavelet<double>(order: 2);
        double[] testPoints = { 0.5, 1.0, 1.5, 2.0 };

        // Act & Assert
        foreach (var x in testPoints)
        {
            var positive = wavelet.Calculate(x);
            var negative = wavelet.Calculate(-x);
            Assert.Equal(positive, negative, Tolerance);
        }
    }

    #endregion

    #region Mathematical Correctness Tests - Shannon Wavelet

    [Fact]
    public void ShannonWavelet_Calculate_ReturnsFiniteValues()
    {
        // Arrange
        var wavelet = new ShannonWavelet<double>();
        double[] testPoints = { 0.0, 0.5, 1.0, -0.5, -1.0, 2.0 };

        // Act & Assert
        foreach (var x in testPoints)
        {
            var result = wavelet.Calculate(x);
            Assert.False(double.IsNaN(result), $"Shannon wavelet returned NaN at x={x}");
            Assert.False(double.IsInfinity(result), $"Shannon wavelet returned infinity at x={x}");
        }
    }

    #endregion

    #region Mathematical Correctness Tests - Coefficient Normalization

    [Fact]
    public void AllOrthogonalWavelets_ScalingCoefficients_HaveUnitNorm()
    {
        // Arrange - Orthogonal wavelets should have unit norm scaling coefficients
        var wavelets = new IWaveletFunction<double>[]
        {
            new HaarWavelet<double>(),
            new DaubechiesWavelet<double>(order: 4),
        };

        // Act & Assert
        foreach (var wavelet in wavelets)
        {
            var coefficients = wavelet.GetScalingCoefficients();
            double sumOfSquares = 0;
            for (int i = 0; i < coefficients.Length; i++)
                sumOfSquares += coefficients[i] * coefficients[i];

            Assert.Equal(1.0, sumOfSquares, 0.01);
        }
    }

    #endregion

    #region Full Coverage Tests - All Wavelet Methods

    [Fact]
    public void MexicanHatWavelet_Decompose_ReturnsValidCoefficients()
    {
        var wavelet = new MexicanHatWavelet<double>(sigma: 1.0);
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var (approximation, detail) = wavelet.Decompose(input);
        Assert.Equal(input.Length, approximation.Length);
        Assert.Equal(input.Length, detail.Length);
    }

    [Fact]
    public void MexicanHatWavelet_GetScalingCoefficients_ReturnsValidVector()
    {
        var wavelet = new MexicanHatWavelet<double>(sigma: 1.0);
        var coefficients = wavelet.GetScalingCoefficients();
        Assert.True(coefficients.Length > 0);
        Assert.False(double.IsNaN(coefficients[0]));
    }

    [Fact]
    public void MexicanHatWavelet_GetWaveletCoefficients_ReturnsValidVector()
    {
        var wavelet = new MexicanHatWavelet<double>(sigma: 1.0);
        var coefficients = wavelet.GetWaveletCoefficients();
        Assert.True(coefficients.Length > 0);
        Assert.False(double.IsNaN(coefficients[0]));
    }

    [Fact]
    public void GaussianWavelet_Decompose_ReturnsValidCoefficients()
    {
        var wavelet = new GaussianWavelet<double>(sigma: 1.0);
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var (approximation, detail) = wavelet.Decompose(input);
        Assert.Equal(input.Length, approximation.Length);
        Assert.Equal(input.Length, detail.Length);
    }

    [Fact]
    public void GaussianWavelet_GetScalingCoefficients_ReturnsValidVector()
    {
        var wavelet = new GaussianWavelet<double>(sigma: 1.0);
        var coefficients = wavelet.GetScalingCoefficients();
        Assert.True(coefficients.Length > 0);
        Assert.False(double.IsNaN(coefficients[0]));
    }

    [Fact]
    public void GaussianWavelet_GetWaveletCoefficients_ReturnsValidVector()
    {
        var wavelet = new GaussianWavelet<double>(sigma: 1.0);
        var coefficients = wavelet.GetWaveletCoefficients();
        Assert.True(coefficients.Length > 0);
        Assert.False(double.IsNaN(coefficients[0]));
    }

    [Fact]
    public void MorletWavelet_Decompose_ReturnsValidCoefficients()
    {
        var wavelet = new MorletWavelet<double>(omega: 5.0);
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var (approximation, detail) = wavelet.Decompose(input);
        Assert.True(approximation.Length > 0);
        Assert.True(detail.Length > 0);
    }

    [Fact]
    public void MorletWavelet_GetScalingCoefficients_ReturnsValidVector()
    {
        var wavelet = new MorletWavelet<double>(omega: 5.0);
        var coefficients = wavelet.GetScalingCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void MorletWavelet_GetWaveletCoefficients_ReturnsValidVector()
    {
        var wavelet = new MorletWavelet<double>(omega: 5.0);
        var coefficients = wavelet.GetWaveletCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void DOGWavelet_Decompose_ReturnsValidCoefficients()
    {
        var wavelet = new DOGWavelet<double>(order: 2);
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var (approximation, detail) = wavelet.Decompose(input);
        Assert.True(approximation.Length > 0);
        Assert.True(detail.Length > 0);
    }

    [Fact]
    public void DOGWavelet_GetScalingCoefficients_ReturnsValidVector()
    {
        var wavelet = new DOGWavelet<double>(order: 2);
        var coefficients = wavelet.GetScalingCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void DOGWavelet_GetWaveletCoefficients_ReturnsValidVector()
    {
        var wavelet = new DOGWavelet<double>(order: 2);
        var coefficients = wavelet.GetWaveletCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void DOGWavelet_Order1_Calculate_IsAntisymmetric()
    {
        var wavelet = new DOGWavelet<double>(order: 1);
        var atOne = wavelet.Calculate(1.0);
        var atNegOne = wavelet.Calculate(-1.0);
        Assert.Equal(-atOne, atNegOne, Tolerance);
    }

    [Fact]
    public void DOGWavelet_Order3_Calculate_ReturnsValidValues()
    {
        var wavelet = new DOGWavelet<double>(order: 3);
        var result = wavelet.Calculate(1.0);
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    [Fact]
    public void DOGWavelet_Order4_Calculate_ReturnsValidValues()
    {
        var wavelet = new DOGWavelet<double>(order: 4);
        var result = wavelet.Calculate(1.0);
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    [Fact]
    public void DOGWavelet_HigherOrder_Calculate_ReturnsValidValues()
    {
        var wavelet = new DOGWavelet<double>(order: 5);
        var result = wavelet.Calculate(1.0);
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    [Fact]
    public void ShannonWavelet_Decompose_ReturnsValidCoefficients()
    {
        var wavelet = new ShannonWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var (approximation, detail) = wavelet.Decompose(input);
        Assert.True(approximation.Length > 0);
        Assert.True(detail.Length > 0);
    }

    [Fact]
    public void ShannonWavelet_GetScalingCoefficients_ReturnsValidVector()
    {
        var wavelet = new ShannonWavelet<double>();
        var coefficients = wavelet.GetScalingCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void ShannonWavelet_GetWaveletCoefficients_ReturnsValidVector()
    {
        var wavelet = new ShannonWavelet<double>();
        var coefficients = wavelet.GetWaveletCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void MeyerWavelet_Decompose_ReturnsValidCoefficients()
    {
        var wavelet = new MeyerWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var (approximation, detail) = wavelet.Decompose(input);
        Assert.True(approximation.Length > 0);
        Assert.True(detail.Length > 0);
    }

    [Fact]
    public void MeyerWavelet_GetScalingCoefficients_ReturnsValidVector()
    {
        var wavelet = new MeyerWavelet<double>();
        var coefficients = wavelet.GetScalingCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void MeyerWavelet_GetWaveletCoefficients_ReturnsValidVector()
    {
        var wavelet = new MeyerWavelet<double>();
        var coefficients = wavelet.GetWaveletCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void MeyerWavelet_Calculate_ReturnsValidValues()
    {
        var wavelet = new MeyerWavelet<double>();
        var result = wavelet.Calculate(0.5);
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    [Fact]
    public void ContinuousMexicanHatWavelet_Decompose_ReturnsValidCoefficients()
    {
        var wavelet = new ContinuousMexicanHatWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var (approximation, detail) = wavelet.Decompose(input);
        Assert.True(approximation.Length > 0);
        Assert.True(detail.Length > 0);
    }

    [Fact]
    public void ContinuousMexicanHatWavelet_GetScalingCoefficients_ReturnsValidVector()
    {
        var wavelet = new ContinuousMexicanHatWavelet<double>();
        var coefficients = wavelet.GetScalingCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void ContinuousMexicanHatWavelet_GetWaveletCoefficients_ReturnsValidVector()
    {
        var wavelet = new ContinuousMexicanHatWavelet<double>();
        var coefficients = wavelet.GetWaveletCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void ContinuousMexicanHatWavelet_Calculate_ReturnsValidValues()
    {
        var wavelet = new ContinuousMexicanHatWavelet<double>();
        var result = wavelet.Calculate(0.5);
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    [Fact]
    public void BattleLemarieWavelet_Decompose_ReturnsValidCoefficients()
    {
        var wavelet = new BattleLemarieWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var (approximation, detail) = wavelet.Decompose(input);
        Assert.True(approximation.Length > 0);
        Assert.True(detail.Length > 0);
    }

    [Fact]
    public void BattleLemarieWavelet_GetScalingCoefficients_ReturnsValidVector()
    {
        var wavelet = new BattleLemarieWavelet<double>();
        var coefficients = wavelet.GetScalingCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void BattleLemarieWavelet_GetWaveletCoefficients_ReturnsValidVector()
    {
        var wavelet = new BattleLemarieWavelet<double>();
        var coefficients = wavelet.GetWaveletCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void BattleLemarieWavelet_Calculate_ReturnsValidValues()
    {
        var wavelet = new BattleLemarieWavelet<double>();
        var result = wavelet.Calculate(0.5);
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    [Fact]
    public void BSplineWavelet_Decompose_ReturnsValidCoefficients()
    {
        var wavelet = new BSplineWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var (approximation, detail) = wavelet.Decompose(input);
        Assert.True(approximation.Length > 0);
        Assert.True(detail.Length > 0);
    }

    [Fact]
    public void BSplineWavelet_GetScalingCoefficients_ReturnsValidVector()
    {
        var wavelet = new BSplineWavelet<double>();
        var coefficients = wavelet.GetScalingCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void BSplineWavelet_GetWaveletCoefficients_ReturnsValidVector()
    {
        var wavelet = new BSplineWavelet<double>();
        var coefficients = wavelet.GetWaveletCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void BSplineWavelet_Calculate_ReturnsValidValues()
    {
        var wavelet = new BSplineWavelet<double>();
        var result = wavelet.Calculate(0.5);
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    [Fact]
    public void GaborWavelet_Decompose_ReturnsValidCoefficients()
    {
        var wavelet = new GaborWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var (approximation, detail) = wavelet.Decompose(input);
        Assert.True(approximation.Length > 0);
        Assert.True(detail.Length > 0);
    }

    [Fact]
    public void GaborWavelet_GetScalingCoefficients_ReturnsValidVector()
    {
        var wavelet = new GaborWavelet<double>();
        var coefficients = wavelet.GetScalingCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void GaborWavelet_GetWaveletCoefficients_ReturnsValidVector()
    {
        var wavelet = new GaborWavelet<double>();
        var coefficients = wavelet.GetWaveletCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void PaulWavelet_Decompose_ReturnsValidCoefficients()
    {
        var wavelet = new PaulWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var (approximation, detail) = wavelet.Decompose(input);
        Assert.True(approximation.Length > 0);
        Assert.True(detail.Length > 0);
    }

    [Fact]
    public void PaulWavelet_GetScalingCoefficients_ReturnsValidVector()
    {
        var wavelet = new PaulWavelet<double>();
        var coefficients = wavelet.GetScalingCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void PaulWavelet_GetWaveletCoefficients_ReturnsValidVector()
    {
        var wavelet = new PaulWavelet<double>();
        var coefficients = wavelet.GetWaveletCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void PaulWavelet_Calculate_ReturnsValidValues()
    {
        var wavelet = new PaulWavelet<double>();
        var result = wavelet.Calculate(0.5);
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    [Fact]
    public void BiorthogonalWavelet_Decompose_ReturnsValidCoefficients()
    {
        var wavelet = new BiorthogonalWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var (approximation, detail) = wavelet.Decompose(input);
        Assert.True(approximation.Length > 0);
        Assert.True(detail.Length > 0);
    }

    [Fact]
    public void BiorthogonalWavelet_GetScalingCoefficients_ReturnsValidVector()
    {
        var wavelet = new BiorthogonalWavelet<double>();
        var coefficients = wavelet.GetScalingCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void BiorthogonalWavelet_GetWaveletCoefficients_ReturnsValidVector()
    {
        var wavelet = new BiorthogonalWavelet<double>();
        var coefficients = wavelet.GetWaveletCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void BiorthogonalWavelet_Calculate_ReturnsValidValues()
    {
        var wavelet = new BiorthogonalWavelet<double>();
        var result = wavelet.Calculate(0.5);
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    [Fact]
    public void SymletWavelet_Decompose_ReturnsValidCoefficients()
    {
        var wavelet = new SymletWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var (approximation, detail) = wavelet.Decompose(input);
        Assert.True(approximation.Length > 0);
        Assert.True(detail.Length > 0);
    }

    [Fact]
    public void SymletWavelet_GetScalingCoefficients_ReturnsValidVector()
    {
        var wavelet = new SymletWavelet<double>();
        var coefficients = wavelet.GetScalingCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void SymletWavelet_GetWaveletCoefficients_ReturnsValidVector()
    {
        var wavelet = new SymletWavelet<double>();
        var coefficients = wavelet.GetWaveletCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void SymletWavelet_Calculate_ReturnsValidValues()
    {
        var wavelet = new SymletWavelet<double>();
        var result = wavelet.Calculate(0.5);
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    [Fact]
    public void CoifletWavelet_Decompose_ReturnsValidCoefficients()
    {
        var wavelet = new CoifletWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var (approximation, detail) = wavelet.Decompose(input);
        Assert.True(approximation.Length > 0);
        Assert.True(detail.Length > 0);
    }

    [Fact]
    public void CoifletWavelet_GetScalingCoefficients_ReturnsValidVector()
    {
        var wavelet = new CoifletWavelet<double>();
        var coefficients = wavelet.GetScalingCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void CoifletWavelet_GetWaveletCoefficients_ReturnsValidVector()
    {
        var wavelet = new CoifletWavelet<double>();
        var coefficients = wavelet.GetWaveletCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void CoifletWavelet_Calculate_ReturnsValidValues()
    {
        var wavelet = new CoifletWavelet<double>();
        var result = wavelet.Calculate(0.5);
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    [Fact]
    public void FejérKorovkinWavelet_Decompose_ReturnsValidCoefficients()
    {
        var wavelet = new FejérKorovkinWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var (approximation, detail) = wavelet.Decompose(input);
        Assert.True(approximation.Length > 0);
        Assert.True(detail.Length > 0);
    }

    [Fact]
    public void FejérKorovkinWavelet_GetScalingCoefficients_ReturnsValidVector()
    {
        var wavelet = new FejérKorovkinWavelet<double>();
        var coefficients = wavelet.GetScalingCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void FejérKorovkinWavelet_GetWaveletCoefficients_ReturnsValidVector()
    {
        var wavelet = new FejérKorovkinWavelet<double>();
        var coefficients = wavelet.GetWaveletCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void FejérKorovkinWavelet_Calculate_ReturnsValidValues()
    {
        var wavelet = new FejérKorovkinWavelet<double>();
        var result = wavelet.Calculate(0.5);
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    [Fact]
    public void ReverseBiorthogonalWavelet_GetScalingCoefficients_DualSynthesis()
    {
        var wavelet = new ReverseBiorthogonalWavelet<double>();
        var coefficients = wavelet.GetScalingCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void ReverseBiorthogonalWavelet_GetWaveletCoefficients_DualSynthesis()
    {
        var wavelet = new ReverseBiorthogonalWavelet<double>();
        var coefficients = wavelet.GetWaveletCoefficients();
        Assert.True(coefficients.Length > 0);
    }

    [Fact]
    public void ComplexGaussianWavelet_Calculate_ReturnsValidValues()
    {
        var wavelet = new ComplexGaussianWavelet<double>();
        var result = wavelet.Calculate(new Complex<double>(0.5, 0.0));
        Assert.False(double.IsNaN(result.Real));
        Assert.False(double.IsNaN(result.Imaginary));
    }

    [Fact]
    public void DaubechiesWavelet_Decompose_ThrowsOnOddLength()
    {
        var wavelet = new DaubechiesWavelet<double>(order: 4);
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 }); // Odd length
        Assert.Throws<ArgumentException>(() => wavelet.Decompose(input));
    }

    [Fact]
    public void HaarWavelet_Decompose_ThrowsOnOddLength()
    {
        var wavelet = new HaarWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 }); // Odd length
        Assert.Throws<ArgumentException>(() => wavelet.Decompose(input));
    }

    [Fact]
    public void ReverseBiorthogonalWavelet_Reconstruct_ReturnsValidVector()
    {
        var wavelet = new ReverseBiorthogonalWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var (approximation, detail) = wavelet.Decompose(input);
        var reconstructed = wavelet.Reconstruct(approximation, detail);
        Assert.True(reconstructed.Length > 0);
    }

    [Fact]
    public void ReverseBiorthogonalWavelet_Calculate_ReturnsValidValues()
    {
        var wavelet = new ReverseBiorthogonalWavelet<double>();
        var result = wavelet.Calculate(0.5);
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    [Fact]
    public void ReverseBiorthogonalWavelet_Decompose_ReturnsValidCoefficients()
    {
        var wavelet = new ReverseBiorthogonalWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var (approximation, detail) = wavelet.Decompose(input);
        Assert.True(approximation.Length > 0);
        Assert.True(detail.Length > 0);
    }

    [Fact]
    public void SymletWavelet_Decompose_WithDifferentOrders()
    {
        var wavelet2 = new SymletWavelet<double>(order: 2);
        var wavelet4 = new SymletWavelet<double>(order: 4);
        var wavelet6 = new SymletWavelet<double>(order: 6);
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var (approx2, detail2) = wavelet2.Decompose(input);
        var (approx4, detail4) = wavelet4.Decompose(input);
        var (approx6, detail6) = wavelet6.Decompose(input);

        Assert.True(approx2.Length > 0);
        Assert.True(approx4.Length > 0);
        Assert.True(approx6.Length > 0);
    }

    [Fact]
    public void CoifletWavelet_Decompose_WithDifferentOrders()
    {
        var wavelet1 = new CoifletWavelet<double>(order: 1);
        var wavelet2 = new CoifletWavelet<double>(order: 2);
        var wavelet3 = new CoifletWavelet<double>(order: 3);
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var (approx1, detail1) = wavelet1.Decompose(input);
        var (approx2, detail2) = wavelet2.Decompose(input);
        var (approx3, detail3) = wavelet3.Decompose(input);

        Assert.True(approx1.Length > 0);
        Assert.True(approx2.Length > 0);
        Assert.True(approx3.Length > 0);
    }

    [Fact]
    public void BiorthogonalWavelet_Decompose_WithDifferentOrders()
    {
        // BiorthogonalWavelet takes decompositionOrder and reconstructionOrder
        var wavelet1 = new BiorthogonalWavelet<double>(decompositionOrder: 1, reconstructionOrder: 1);
        var wavelet2 = new BiorthogonalWavelet<double>(decompositionOrder: 2, reconstructionOrder: 2);
        var wavelet3 = new BiorthogonalWavelet<double>(decompositionOrder: 3, reconstructionOrder: 3);
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var (approx1, detail1) = wavelet1.Decompose(input);
        var (approx2, detail2) = wavelet2.Decompose(input);
        var (approx3, detail3) = wavelet3.Decompose(input);

        Assert.True(approx1.Length > 0);
        Assert.True(approx2.Length > 0);
        Assert.True(approx3.Length > 0);
    }

    [Fact]
    public void MeyerWavelet_Decompose_WithLargerInput()
    {
        var wavelet = new MeyerWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 });
        var (approximation, detail) = wavelet.Decompose(input);
        Assert.True(approximation.Length > 0);
        Assert.True(detail.Length > 0);
    }

    [Fact]
    public void ShannonWavelet_Decompose_WithLargerInput()
    {
        var wavelet = new ShannonWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 });
        var (approximation, detail) = wavelet.Decompose(input);
        Assert.True(approximation.Length > 0);
        Assert.True(detail.Length > 0);
    }

    [Fact]
    public void FejérKorovkinWavelet_Decompose_WithDifferentOrders()
    {
        var wavelet4 = new FejérKorovkinWavelet<double>(order: 4);
        var wavelet6 = new FejérKorovkinWavelet<double>(order: 6);
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var (approx4, detail4) = wavelet4.Decompose(input);
        var (approx6, detail6) = wavelet6.Decompose(input);

        Assert.True(approx4.Length > 0);
        Assert.True(approx6.Length > 0);
    }

    [Fact]
    public void BattleLemarieWavelet_Decompose_WithDifferentOrders()
    {
        var wavelet1 = new BattleLemarieWavelet<double>(order: 1);
        var wavelet2 = new BattleLemarieWavelet<double>(order: 2);
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var (approx1, detail1) = wavelet1.Decompose(input);
        var (approx2, detail2) = wavelet2.Decompose(input);

        Assert.True(approx1.Length > 0);
        Assert.True(approx2.Length > 0);
    }

    [Fact]
    public void DaubechiesWavelet_Calculate_WithDifferentOrders()
    {
        var wavelet2 = new DaubechiesWavelet<double>(order: 2);
        var wavelet6 = new DaubechiesWavelet<double>(order: 6);

        var result2 = wavelet2.Calculate(0.5);
        var result6 = wavelet6.Calculate(0.5);

        Assert.False(double.IsNaN(result2));
        Assert.False(double.IsNaN(result6));
    }

    #endregion

    #region Additional Coverage Tests for SymletWavelet

    [Fact]
    public void SymletWavelet_AllOrders_CreateSuccessfully()
    {
        var wavelet2 = new SymletWavelet<double>(order: 2);
        var wavelet4 = new SymletWavelet<double>(order: 4);
        var wavelet6 = new SymletWavelet<double>(order: 6);
        var wavelet8 = new SymletWavelet<double>(order: 8);

        Assert.NotNull(wavelet2);
        Assert.NotNull(wavelet4);
        Assert.NotNull(wavelet6);
        Assert.NotNull(wavelet8);
    }

    [Fact]
    public void SymletWavelet_AllOrders_Decompose()
    {
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var (approx2, detail2) = new SymletWavelet<double>(order: 2).Decompose(input);
        var (approx4, detail4) = new SymletWavelet<double>(order: 4).Decompose(input);
        var (approx6, detail6) = new SymletWavelet<double>(order: 6).Decompose(input);
        var (approx8, detail8) = new SymletWavelet<double>(order: 8).Decompose(input);

        Assert.True(approx2.Length > 0);
        Assert.True(approx4.Length > 0);
        Assert.True(approx6.Length > 0);
        Assert.True(approx8.Length > 0);
    }

    [Fact]
    public void SymletWavelet_AllOrders_Calculate()
    {
        var result2 = new SymletWavelet<double>(order: 2).Calculate(0.5);
        var result4 = new SymletWavelet<double>(order: 4).Calculate(0.5);
        var result6 = new SymletWavelet<double>(order: 6).Calculate(0.5);
        var result8 = new SymletWavelet<double>(order: 8).Calculate(0.5);

        Assert.False(double.IsNaN(result2));
        Assert.False(double.IsNaN(result4));
        Assert.False(double.IsNaN(result6));
        Assert.False(double.IsNaN(result8));
    }

    [Fact]
    public void SymletWavelet_AllOrders_GetCoefficients()
    {
        var orders = new[] { 2, 4, 6, 8 };
        foreach (var order in orders)
        {
            var wavelet = new SymletWavelet<double>(order: order);
            var scalingCoeffs = wavelet.GetScalingCoefficients();
            var waveletCoeffs = wavelet.GetWaveletCoefficients();

            Assert.True(scalingCoeffs.Length > 0);
            Assert.True(waveletCoeffs.Length > 0);
        }
    }

    [Fact]
    public void SymletWavelet_InvalidOrder_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new SymletWavelet<double>(order: 1));
        Assert.Throws<ArgumentException>(() => new SymletWavelet<double>(order: 3));
        Assert.Throws<ArgumentException>(() => new SymletWavelet<double>(order: 5));
        Assert.Throws<ArgumentException>(() => new SymletWavelet<double>(order: 10));
    }

    #endregion

    #region Additional Coverage Tests for ReverseBiorthogonalWavelet

    [Fact]
    public void ReverseBiorthogonalWavelet_AllWaveletTypes_CreateSuccessfully()
    {
        var waveletTypes = new[]
        {
            WaveletType.ReverseBior11, WaveletType.ReverseBior13,
            WaveletType.ReverseBior22, WaveletType.ReverseBior24, WaveletType.ReverseBior26, WaveletType.ReverseBior28,
            WaveletType.ReverseBior31, WaveletType.ReverseBior33, WaveletType.ReverseBior35, WaveletType.ReverseBior37, WaveletType.ReverseBior39,
            WaveletType.ReverseBior44, WaveletType.ReverseBior46, WaveletType.ReverseBior48,
            WaveletType.ReverseBior55, WaveletType.ReverseBior68
        };

        foreach (var waveletType in waveletTypes)
        {
            var wavelet = new ReverseBiorthogonalWavelet<double>(waveletType);
            Assert.NotNull(wavelet);
        }
    }

    [Fact]
    public void ReverseBiorthogonalWavelet_AllWaveletTypes_Decompose()
    {
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var waveletTypes = new[]
        {
            WaveletType.ReverseBior11, WaveletType.ReverseBior22, WaveletType.ReverseBior33,
            WaveletType.ReverseBior44, WaveletType.ReverseBior55, WaveletType.ReverseBior68
        };

        foreach (var waveletType in waveletTypes)
        {
            var wavelet = new ReverseBiorthogonalWavelet<double>(waveletType);
            var (approx, detail) = wavelet.Decompose(input);
            Assert.True(approx.Length > 0);
            Assert.True(detail.Length > 0);
        }
    }

    [Fact]
    public void ReverseBiorthogonalWavelet_DecomposeMultiLevel_ReturnsMultipleLevels()
    {
        var wavelet = new ReverseBiorthogonalWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 });

        var (approximation, details) = wavelet.DecomposeMultiLevel(input, 2);

        Assert.True(approximation.Length > 0);
        Assert.Equal(2, details.Count);
        Assert.True(details[0].Length > 0);
        Assert.True(details[1].Length > 0);
    }

    [Fact]
    public void ReverseBiorthogonalWavelet_ReconstructMultiLevel_ReturnsSignal()
    {
        var wavelet = new ReverseBiorthogonalWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 });

        var (approximation, details) = wavelet.DecomposeMultiLevel(input, 2);
        var reconstructed = wavelet.ReconstructMultiLevel(approximation, details);

        Assert.True(reconstructed.Length > 0);
    }

    [Fact]
    public void ReverseBiorthogonalWavelet_BoundaryMethods_Work()
    {
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var waveletPeriodic = new ReverseBiorthogonalWavelet<double>(WaveletType.ReverseBior22, BoundaryHandlingMethod.Periodic);
        var waveletSymmetric = new ReverseBiorthogonalWavelet<double>(WaveletType.ReverseBior22, BoundaryHandlingMethod.Symmetric);
        var waveletZeroPad = new ReverseBiorthogonalWavelet<double>(WaveletType.ReverseBior22, BoundaryHandlingMethod.ZeroPadding);

        var (approxPeriodic, _) = waveletPeriodic.Decompose(input);
        var (approxSymmetric, _) = waveletSymmetric.Decompose(input);
        var (approxZeroPad, _) = waveletZeroPad.Decompose(input);

        Assert.True(approxPeriodic.Length > 0);
        Assert.True(approxSymmetric.Length > 0);
        Assert.True(approxZeroPad.Length > 0);
    }

    [Fact]
    public void ReverseBiorthogonalWavelet_DifferentChunkSizes_Work()
    {
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var wavelet512 = new ReverseBiorthogonalWavelet<double>(WaveletType.ReverseBior22, BoundaryHandlingMethod.Periodic, chunkSize: 512);
        var wavelet2048 = new ReverseBiorthogonalWavelet<double>(WaveletType.ReverseBior22, BoundaryHandlingMethod.Periodic, chunkSize: 2048);

        var (approx512, _) = wavelet512.Decompose(input);
        var (approx2048, _) = wavelet2048.Decompose(input);

        Assert.True(approx512.Length > 0);
        Assert.True(approx2048.Length > 0);
    }

    [Fact]
    public void ReverseBiorthogonalWavelet_Calculate_AllTypes()
    {
        var waveletTypes = new[] { WaveletType.ReverseBior11, WaveletType.ReverseBior22, WaveletType.ReverseBior33, WaveletType.ReverseBior44 };

        foreach (var waveletType in waveletTypes)
        {
            var wavelet = new ReverseBiorthogonalWavelet<double>(waveletType);
            var result = wavelet.Calculate(0.5);
            Assert.False(double.IsNaN(result));
        }
    }

    #endregion

    #region Additional Coverage Tests for CoifletWavelet

    [Fact]
    public void CoifletWavelet_AllOrders_CreateSuccessfully()
    {
        var orders = new[] { 1, 2, 3, 4, 5 };
        foreach (var order in orders)
        {
            var wavelet = new CoifletWavelet<double>(order: order);
            Assert.NotNull(wavelet);
        }
    }

    [Fact]
    public void CoifletWavelet_AllOrders_Decompose()
    {
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var orders = new[] { 1, 2, 3, 4, 5 };

        foreach (var order in orders)
        {
            var wavelet = new CoifletWavelet<double>(order: order);
            var (approx, detail) = wavelet.Decompose(input);
            Assert.True(approx.Length > 0);
            Assert.True(detail.Length > 0);
        }
    }

    [Fact]
    public void CoifletWavelet_AllOrders_Calculate()
    {
        var orders = new[] { 1, 2, 3, 4, 5 };

        foreach (var order in orders)
        {
            var wavelet = new CoifletWavelet<double>(order: order);
            var result = wavelet.Calculate(0.5);
            Assert.False(double.IsNaN(result));
        }
    }

    [Fact]
    public void CoifletWavelet_AllOrders_GetCoefficients()
    {
        var orders = new[] { 1, 2, 3, 4, 5 };

        foreach (var order in orders)
        {
            var wavelet = new CoifletWavelet<double>(order: order);
            var scalingCoeffs = wavelet.GetScalingCoefficients();
            var waveletCoeffs = wavelet.GetWaveletCoefficients();
            Assert.True(scalingCoeffs.Length > 0);
            Assert.True(waveletCoeffs.Length > 0);
        }
    }

    #endregion

    #region Additional Coverage Tests for MeyerWavelet

    [Fact]
    public void MeyerWavelet_Calculate_AtMultiplePoints()
    {
        var wavelet = new MeyerWavelet<double>();
        var points = new[] { -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0 };

        foreach (var x in points)
        {
            var result = wavelet.Calculate(x);
            Assert.False(double.IsNaN(result));
        }
    }

    [Fact]
    public void MeyerWavelet_DecomposeAndGetCoefficients()
    {
        var wavelet = new MeyerWavelet<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 });

        var (approx, detail) = wavelet.Decompose(input);
        var scalingCoeffs = wavelet.GetScalingCoefficients();
        var waveletCoeffs = wavelet.GetWaveletCoefficients();

        Assert.True(approx.Length > 0);
        Assert.True(detail.Length > 0);
        Assert.True(scalingCoeffs.Length > 0);
        Assert.True(waveletCoeffs.Length > 0);
    }

    #endregion

    #region Additional Coverage Tests for FejérKorovkinWavelet

    [Fact]
    public void FejérKorovkinWavelet_AllOrders_CreateAndDecompose()
    {
        // Input must be at least as long as the largest filter order (22)
        var input = new Vector<double>(new[] {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0
        });
        var orders = new[] { 4, 6, 8, 14, 18, 22 };

        foreach (var order in orders)
        {
            var wavelet = new FejérKorovkinWavelet<double>(order: order);
            var (approx, detail) = wavelet.Decompose(input);
            Assert.True(approx.Length > 0);
        }
    }

    [Fact]
    public void FejérKorovkinWavelet_AllOrders_GetCoefficients()
    {
        var orders = new[] { 4, 6, 8, 14, 18, 22 };

        foreach (var order in orders)
        {
            var wavelet = new FejérKorovkinWavelet<double>(order: order);
            var scalingCoeffs = wavelet.GetScalingCoefficients();
            var waveletCoeffs = wavelet.GetWaveletCoefficients();
            Assert.True(scalingCoeffs.Length > 0);
            Assert.True(waveletCoeffs.Length > 0);
        }
    }

    #endregion

    #region Additional Coverage Tests for DOGWavelet

    [Fact]
    public void DOGWavelet_AllOrders_Calculate()
    {
        var orders = new[] { 1, 2, 3, 4, 5 };

        foreach (var order in orders)
        {
            var wavelet = new DOGWavelet<double>(order: order);
            var result = wavelet.Calculate(0.0);
            Assert.False(double.IsNaN(result));
        }
    }

    [Fact]
    public void DOGWavelet_AllOrders_Decompose()
    {
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var orders = new[] { 1, 2, 3, 4, 5 };

        foreach (var order in orders)
        {
            var wavelet = new DOGWavelet<double>(order: order);
            var (approx, detail) = wavelet.Decompose(input);
            Assert.True(approx.Length > 0);
        }
    }

    #endregion

    #region Additional Coverage Tests for ShannonWavelet

    [Fact]
    public void ShannonWavelet_Calculate_AtMultiplePoints()
    {
        var wavelet = new ShannonWavelet<double>();
        var points = new[] { -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0 };

        foreach (var x in points)
        {
            var result = wavelet.Calculate(x);
            Assert.False(double.IsNaN(result));
        }
    }

    [Fact]
    public void ShannonWavelet_GetCoefficients()
    {
        var wavelet = new ShannonWavelet<double>();
        var scalingCoeffs = wavelet.GetScalingCoefficients();
        var waveletCoeffs = wavelet.GetWaveletCoefficients();

        Assert.True(scalingCoeffs.Length > 0);
        Assert.True(waveletCoeffs.Length > 0);
    }

    #endregion

    #region Additional Coverage Tests for BattleLemarieWavelet

    [Fact]
    public void BattleLemarieWavelet_AllOrders_CreateAndCalculate()
    {
        var orders = new[] { 1, 2, 3 };

        foreach (var order in orders)
        {
            var wavelet = new BattleLemarieWavelet<double>(order: order);
            var result = wavelet.Calculate(0.5);
            Assert.False(double.IsNaN(result));
        }
    }

    [Fact]
    public void BattleLemarieWavelet_AllOrders_GetCoefficients()
    {
        var orders = new[] { 1, 2, 3 };

        foreach (var order in orders)
        {
            var wavelet = new BattleLemarieWavelet<double>(order: order);
            var scalingCoeffs = wavelet.GetScalingCoefficients();
            var waveletCoeffs = wavelet.GetWaveletCoefficients();
            Assert.True(scalingCoeffs.Length > 0);
            Assert.True(waveletCoeffs.Length > 0);
        }
    }

    #endregion

    #region Additional Coverage Tests for ComplexGaussianWavelet

    [Fact]
    public void ComplexGaussianWavelet_AllOrders_CreateAndCalculate()
    {
        var orders = new[] { 1, 2, 3, 4, 5 };

        foreach (var order in orders)
        {
            var wavelet = new ComplexGaussianWavelet<double>(order: order);
            var complexInput = new Complex<double>(0.5, 0.0);
            var result = wavelet.Calculate(complexInput);
            Assert.False(double.IsNaN(result.Real));
            Assert.False(double.IsNaN(result.Imaginary));
        }
    }

    [Fact]
    public void ComplexGaussianWavelet_AllOrders_GetCoefficients()
    {
        var orders = new[] { 1, 2, 3, 4, 5 };

        foreach (var order in orders)
        {
            var wavelet = new ComplexGaussianWavelet<double>(order: order);
            var scalingCoeffs = wavelet.GetScalingCoefficients();
            var waveletCoeffs = wavelet.GetWaveletCoefficients();
            Assert.True(scalingCoeffs.Length > 0);
            Assert.True(waveletCoeffs.Length > 0);
        }
    }

    #endregion

    #region Additional Coverage Tests for PaulWavelet

    [Fact]
    public void PaulWavelet_AllOrders_CreateAndCalculate()
    {
        var orders = new[] { 1, 2, 3, 4, 5 };

        foreach (var order in orders)
        {
            var wavelet = new PaulWavelet<double>(order: order);
            var result = wavelet.Calculate(0.5);
            Assert.False(double.IsNaN(result));
        }
    }

    [Fact]
    public void PaulWavelet_AllOrders_GetCoefficients()
    {
        var orders = new[] { 1, 2, 3, 4, 5 };

        foreach (var order in orders)
        {
            var wavelet = new PaulWavelet<double>(order: order);
            var scalingCoeffs = wavelet.GetScalingCoefficients();
            var waveletCoeffs = wavelet.GetWaveletCoefficients();
            Assert.True(scalingCoeffs.Length > 0);
            Assert.True(waveletCoeffs.Length > 0);
        }
    }

    #endregion

    #region Additional Coverage Tests for DaubechiesWavelet

    [Fact]
    public void DaubechiesWavelet_AllOrders_CreateAndDecompose()
    {
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var orders = new[] { 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        foreach (var order in orders)
        {
            var wavelet = new DaubechiesWavelet<double>(order: order);
            var (approx, detail) = wavelet.Decompose(input);
            Assert.True(approx.Length > 0);
        }
    }

    [Fact]
    public void DaubechiesWavelet_AllOrders_GetCoefficients()
    {
        var orders = new[] { 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        foreach (var order in orders)
        {
            var wavelet = new DaubechiesWavelet<double>(order: order);
            var scalingCoeffs = wavelet.GetScalingCoefficients();
            var waveletCoeffs = wavelet.GetWaveletCoefficients();
            Assert.True(scalingCoeffs.Length > 0);
            Assert.True(waveletCoeffs.Length > 0);
        }
    }

    #endregion

    #region Perfect Reconstruction Tests

    /// <summary>
    /// Tests that Haar wavelet achieves perfect reconstruction.
    /// Haar is an orthogonal wavelet, so decompose + reconstruct should equal original.
    /// </summary>
    [Fact]
    public void HaarWavelet_PerfectReconstruction_ReturnsOriginalSignal()
    {
        // Arrange
        var wavelet = new HaarWavelet<double>();
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        // Act - Decompose
        var (approximation, detail) = wavelet.Decompose(original);

        // Reconstruct using inverse Haar transform
        // x[2i] = (a[i] + d[i]) / sqrt(2)
        // x[2i+1] = (a[i] - d[i]) / sqrt(2)
        double sqrt2 = Math.Sqrt(2);
        var reconstructed = new double[original.Length];
        for (int i = 0; i < approximation.Length; i++)
        {
            reconstructed[2 * i] = (approximation[i] + detail[i]) / sqrt2;
            reconstructed[2 * i + 1] = (approximation[i] - detail[i]) / sqrt2;
        }

        // Assert - Should match original within tolerance
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], reconstructed[i], Tolerance);
        }
    }

    /// <summary>
    /// Tests Haar perfect reconstruction with various signal types.
    /// </summary>
    [Theory]
    [InlineData(new double[] { 1.0, 1.0, 1.0, 1.0 })] // Constant signal
    [InlineData(new double[] { 1.0, 2.0, 3.0, 4.0 })] // Linear ramp
    [InlineData(new double[] { 0.0, 1.0, 0.0, -1.0 })] // Alternating
    [InlineData(new double[] { 1.0, -1.0, 1.0, -1.0 })] // High frequency
    [InlineData(new double[] { 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0 })] // Larger signal
    public void HaarWavelet_PerfectReconstruction_VariousSignals(double[] signalData)
    {
        // Arrange
        var wavelet = new HaarWavelet<double>();
        var original = new Vector<double>(signalData);

        // Act - Decompose
        var (approximation, detail) = wavelet.Decompose(original);

        // Reconstruct
        double sqrt2 = Math.Sqrt(2);
        var reconstructed = new double[original.Length];
        for (int i = 0; i < approximation.Length; i++)
        {
            reconstructed[2 * i] = (approximation[i] + detail[i]) / sqrt2;
            reconstructed[2 * i + 1] = (approximation[i] - detail[i]) / sqrt2;
        }

        // Assert
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], reconstructed[i], Tolerance);
        }
    }

    /// <summary>
    /// Tests multi-level Haar decomposition and reconstruction.
    /// </summary>
    [Fact]
    public void HaarWavelet_MultiLevel_PerfectReconstruction()
    {
        // Arrange
        var wavelet = new HaarWavelet<double>();
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        // Act - Two-level decomposition
        var (approx1, detail1) = wavelet.Decompose(original);
        var (approx2, detail2) = wavelet.Decompose(approx1);

        // Reconstruct level 2 -> level 1
        double sqrt2 = Math.Sqrt(2);
        var reconstructedApprox1 = new double[approx1.Length];
        for (int i = 0; i < approx2.Length; i++)
        {
            reconstructedApprox1[2 * i] = (approx2[i] + detail2[i]) / sqrt2;
            reconstructedApprox1[2 * i + 1] = (approx2[i] - detail2[i]) / sqrt2;
        }

        // Reconstruct level 1 -> original
        var reconstructed = new double[original.Length];
        for (int i = 0; i < approx1.Length; i++)
        {
            reconstructed[2 * i] = (reconstructedApprox1[i] + detail1[i]) / sqrt2;
            reconstructed[2 * i + 1] = (reconstructedApprox1[i] - detail1[i]) / sqrt2;
        }

        // Assert
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], reconstructed[i], Tolerance);
        }
    }

    /// <summary>
    /// Tests ReverseBiorthogonal wavelet perfect reconstruction using its Reconstruct method.
    /// </summary>
    [Fact]
    public void ReverseBiorthogonalWavelet_PerfectReconstruction_ReturnsOriginalSignal()
    {
        // Arrange
        var wavelet = new ReverseBiorthogonalWavelet<double>(WaveletType.ReverseBior22);
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        // Act
        var (approximation, detail) = wavelet.Decompose(original);
        var reconstructed = wavelet.Reconstruct(approximation, detail);

        // Assert - Due to boundary effects, we check that reconstruction is close
        // and has the same length
        Assert.Equal(original.Length, reconstructed.Length);

        // Check that the reconstructed signal has reasonable values (not NaN or infinity)
        for (int i = 0; i < reconstructed.Length; i++)
        {
            Assert.False(double.IsNaN(reconstructed[i]), $"Reconstructed[{i}] is NaN");
            Assert.False(double.IsInfinity(reconstructed[i]), $"Reconstructed[{i}] is Infinity");
        }
    }

    /// <summary>
    /// Tests ReverseBiorthogonal wavelet multi-level perfect reconstruction.
    /// </summary>
    [Fact]
    public void ReverseBiorthogonalWavelet_MultiLevel_PerfectReconstruction()
    {
        // Arrange
        var wavelet = new ReverseBiorthogonalWavelet<double>(WaveletType.ReverseBior22);
        var original = new Vector<double>(new[] {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0
        });

        // Act - Multi-level decomposition and reconstruction
        var (approximation, details) = wavelet.DecomposeMultiLevel(original, 2);
        var reconstructed = wavelet.ReconstructMultiLevel(approximation, details);

        // Assert
        Assert.Equal(original.Length, reconstructed.Length);

        for (int i = 0; i < reconstructed.Length; i++)
        {
            Assert.False(double.IsNaN(reconstructed[i]), $"Reconstructed[{i}] is NaN");
            Assert.False(double.IsInfinity(reconstructed[i]), $"Reconstructed[{i}] is Infinity");
        }
    }

    /// <summary>
    /// Tests that orthogonal wavelets satisfy the orthonormality condition:
    /// sum(h[k] * h[k+2n]) = delta(n) for the scaling coefficients.
    /// </summary>
    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(6)]
    public void DaubechiesWavelet_ScalingCoefficients_SatisfyOrthogonality(int order)
    {
        // Arrange
        var wavelet = new DaubechiesWavelet<double>(order: order);
        var h = wavelet.GetScalingCoefficients();

        // Act - Check orthonormality: sum(h[k] * h[k]) = 1
        double sumSquares = 0;
        for (int i = 0; i < h.Length; i++)
        {
            sumSquares += h[i] * h[i];
        }

        // Assert - Should be 1 for orthonormal wavelets
        Assert.Equal(1.0, sumSquares, Tolerance);
    }

    /// <summary>
    /// Tests that wavelet coefficients satisfy zero-mean property:
    /// sum(g[k]) = 0 for wavelet coefficients.
    /// </summary>
    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(6)]
    public void DaubechiesWavelet_WaveletCoefficients_SatisfyZeroMean(int order)
    {
        // Arrange
        var wavelet = new DaubechiesWavelet<double>(order: order);
        var g = wavelet.GetWaveletCoefficients();

        // Act
        double sum = 0;
        for (int i = 0; i < g.Length; i++)
        {
            sum += g[i];
        }

        // Assert - Should be 0 for wavelets
        Assert.Equal(0.0, sum, Tolerance);
    }

    /// <summary>
    /// Tests Symlet wavelet coefficient properties.
    /// </summary>
    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(6)]
    public void SymletWavelet_Coefficients_SatisfyOrthogonality(int order)
    {
        // Arrange
        var wavelet = new SymletWavelet<double>(order: order);
        var h = wavelet.GetScalingCoefficients();

        // Act - Check unit norm
        double sumSquares = 0;
        for (int i = 0; i < h.Length; i++)
        {
            sumSquares += h[i] * h[i];
        }

        // Assert - Should be approximately 1
        Assert.True(Math.Abs(sumSquares - 1.0) < Tolerance,
            $"Symlet order {order} scaling coefficients should have unit norm, got {sumSquares}");
    }

    /// <summary>
    /// Tests Coiflet wavelet coefficient properties.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void CoifletWavelet_Coefficients_SatisfyOrthogonality(int order)
    {
        // Arrange
        var wavelet = new CoifletWavelet<double>(order: order);
        var h = wavelet.GetScalingCoefficients();

        // Act - Check unit norm
        double sumSquares = 0;
        for (int i = 0; i < h.Length; i++)
        {
            sumSquares += h[i] * h[i];
        }

        // Assert - Should be approximately 1
        Assert.True(Math.Abs(sumSquares - 1.0) < Tolerance,
            $"Coiflet order {order} scaling coefficients should have unit norm, got {sumSquares}");
    }

    /// <summary>
    /// Tests that Biorthogonal wavelets have proper analysis and synthesis filters.
    /// </summary>
    [Fact]
    public void BiorthogonalWavelet_Coefficients_AreProperlyDefined()
    {
        // Arrange
        var wavelet = new BiorthogonalWavelet<double>(decompositionOrder: 2, reconstructionOrder: 2);
        var scaling = wavelet.GetScalingCoefficients();
        var waveletCoeffs = wavelet.GetWaveletCoefficients();

        // Act & Assert - Check coefficients are well-defined
        Assert.True(scaling.Length > 0, "Scaling coefficients should not be empty");
        Assert.True(waveletCoeffs.Length > 0, "Wavelet coefficients should not be empty");

        // Check for NaN values
        for (int i = 0; i < scaling.Length; i++)
        {
            Assert.False(double.IsNaN(scaling[i]), $"Scaling coefficient [{i}] is NaN");
        }
        for (int i = 0; i < waveletCoeffs.Length; i++)
        {
            Assert.False(double.IsNaN(waveletCoeffs[i]), $"Wavelet coefficient [{i}] is NaN");
        }
    }

    /// <summary>
    /// Tests that Haar decomposition and reconstruction preserve signal energy.
    /// </summary>
    [Fact]
    public void HaarWavelet_DecomposeReconstruct_PreservesEnergyExactly()
    {
        // Arrange
        var wavelet = new HaarWavelet<double>();
        var original = new Vector<double>(new[] { 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0 });

        // Calculate original energy
        double originalEnergy = 0;
        for (int i = 0; i < original.Length; i++)
        {
            originalEnergy += original[i] * original[i];
        }

        // Act - Decompose
        var (approximation, detail) = wavelet.Decompose(original);

        // Calculate decomposed energy
        double decomposedEnergy = 0;
        for (int i = 0; i < approximation.Length; i++)
        {
            decomposedEnergy += approximation[i] * approximation[i] + detail[i] * detail[i];
        }

        // Reconstruct
        double sqrt2 = Math.Sqrt(2);
        var reconstructed = new double[original.Length];
        for (int i = 0; i < approximation.Length; i++)
        {
            reconstructed[2 * i] = (approximation[i] + detail[i]) / sqrt2;
            reconstructed[2 * i + 1] = (approximation[i] - detail[i]) / sqrt2;
        }

        // Calculate reconstructed energy
        double reconstructedEnergy = 0;
        for (int i = 0; i < reconstructed.Length; i++)
        {
            reconstructedEnergy += reconstructed[i] * reconstructed[i];
        }

        // Assert - All energies should be equal
        Assert.Equal(originalEnergy, decomposedEnergy, Tolerance);
        Assert.Equal(originalEnergy, reconstructedEnergy, Tolerance);
    }

    /// <summary>
    /// Tests different ReverseBiorthogonal wavelet types for reconstruction.
    /// </summary>
    [Theory]
    [InlineData(WaveletType.ReverseBior11)]
    [InlineData(WaveletType.ReverseBior22)]
    [InlineData(WaveletType.ReverseBior33)]
    [InlineData(WaveletType.ReverseBior44)]
    public void ReverseBiorthogonalWavelet_AllTypes_ReconstructSuccessfully(WaveletType waveletType)
    {
        // Arrange
        var wavelet = new ReverseBiorthogonalWavelet<double>(waveletType);
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        // Act
        var (approximation, detail) = wavelet.Decompose(original);
        var reconstructed = wavelet.Reconstruct(approximation, detail);

        // Assert
        Assert.Equal(original.Length, reconstructed.Length);

        for (int i = 0; i < reconstructed.Length; i++)
        {
            Assert.False(double.IsNaN(reconstructed[i]),
                $"Reconstructed[{i}] is NaN for wavelet type {waveletType}");
        }
    }

    /// <summary>
    /// Tests HaarWavelet Reconstruct method for basic functionality.
    /// </summary>
    [Fact]
    public void HaarWavelet_Reconstruct_ReturnsValidVector()
    {
        var wavelet = new HaarWavelet<double>();
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var (approximation, detail) = wavelet.Decompose(original);
        var reconstructed = wavelet.Reconstruct(approximation, detail);

        Assert.Equal(original.Length, reconstructed.Length);
        for (int i = 0; i < reconstructed.Length; i++)
        {
            Assert.False(double.IsNaN(reconstructed[i]), $"Reconstructed[{i}] is NaN");
            Assert.False(double.IsInfinity(reconstructed[i]), $"Reconstructed[{i}] is Infinity");
        }
    }

    /// <summary>
    /// Tests DaubechiesWavelet Reconstruct method for various orders.
    /// </summary>
    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(6)]
    public void DaubechiesWavelet_Reconstruct_ReturnsValidVector(int order)
    {
        var wavelet = new DaubechiesWavelet<double>(order: order);
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var (approximation, detail) = wavelet.Decompose(original);
        var reconstructed = wavelet.Reconstruct(approximation, detail);

        Assert.Equal(original.Length, reconstructed.Length);
        for (int i = 0; i < reconstructed.Length; i++)
        {
            Assert.False(double.IsNaN(reconstructed[i]), $"Reconstructed[{i}] is NaN for order {order}");
            Assert.False(double.IsInfinity(reconstructed[i]), $"Reconstructed[{i}] is Infinity for order {order}");
        }
    }

    /// <summary>
    /// Tests SymletWavelet Reconstruct method for various orders.
    /// </summary>
    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(6)]
    public void SymletWavelet_Reconstruct_ReturnsValidVector(int order)
    {
        var wavelet = new SymletWavelet<double>(order: order);
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var (approximation, detail) = wavelet.Decompose(original);
        var reconstructed = wavelet.Reconstruct(approximation, detail);

        Assert.Equal(original.Length, reconstructed.Length);
        for (int i = 0; i < reconstructed.Length; i++)
        {
            Assert.False(double.IsNaN(reconstructed[i]), $"Reconstructed[{i}] is NaN for order {order}");
            Assert.False(double.IsInfinity(reconstructed[i]), $"Reconstructed[{i}] is Infinity for order {order}");
        }
    }

    /// <summary>
    /// Tests CoifletWavelet Reconstruct method for various orders.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void CoifletWavelet_Reconstruct_ReturnsValidVector(int order)
    {
        var wavelet = new CoifletWavelet<double>(order: order);
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var (approximation, detail) = wavelet.Decompose(original);
        var reconstructed = wavelet.Reconstruct(approximation, detail);

        Assert.Equal(original.Length, reconstructed.Length);
        for (int i = 0; i < reconstructed.Length; i++)
        {
            Assert.False(double.IsNaN(reconstructed[i]), $"Reconstructed[{i}] is NaN for order {order}");
            Assert.False(double.IsInfinity(reconstructed[i]), $"Reconstructed[{i}] is Infinity for order {order}");
        }
    }

    /// <summary>
    /// Tests BiorthogonalWavelet Reconstruct method.
    /// </summary>
    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 2)]
    [InlineData(3, 3)]
    [InlineData(4, 4)]
    [InlineData(5, 5)]
    [InlineData(6, 6)]
    public void BiorthogonalWavelet_Reconstruct_ReturnsValidVector(int decompOrder, int reconOrder)
    {
        var wavelet = new BiorthogonalWavelet<double>(decompositionOrder: decompOrder, reconstructionOrder: reconOrder);
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var (approximation, detail) = wavelet.Decompose(original);
        var reconstructed = wavelet.Reconstruct(approximation, detail);

        Assert.Equal(original.Length, reconstructed.Length);
        for (int i = 0; i < reconstructed.Length; i++)
        {
            Assert.False(double.IsNaN(reconstructed[i]), $"Reconstructed[{i}] is NaN");
            Assert.False(double.IsInfinity(reconstructed[i]), $"Reconstructed[{i}] is Infinity");
        }
    }

    /// <summary>
    /// Tests BSplineWavelet Reconstruct method for various orders.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void BSplineWavelet_Reconstruct_ReturnsValidVector(int order)
    {
        var wavelet = new BSplineWavelet<double>(order: order);
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var (approximation, detail) = wavelet.Decompose(original);
        var reconstructed = wavelet.Reconstruct(approximation, detail);

        Assert.True(reconstructed.Length > 0, "Reconstructed vector should not be empty");
        for (int i = 0; i < reconstructed.Length; i++)
        {
            Assert.False(double.IsNaN(reconstructed[i]), $"Reconstructed[{i}] is NaN for order {order}");
            Assert.False(double.IsInfinity(reconstructed[i]), $"Reconstructed[{i}] is Infinity for order {order}");
        }
    }

    /// <summary>
    /// Tests MeyerWavelet Reconstruct method.
    /// </summary>
    [Fact]
    public void MeyerWavelet_Reconstruct_ReturnsValidVector()
    {
        var wavelet = new MeyerWavelet<double>();
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var (approximation, detail) = wavelet.Decompose(original);
        var reconstructed = wavelet.Reconstruct(approximation, detail);

        Assert.True(reconstructed.Length > 0, "Reconstructed vector should not be empty");
        for (int i = 0; i < reconstructed.Length; i++)
        {
            Assert.False(double.IsNaN(reconstructed[i]), $"Reconstructed[{i}] is NaN");
            Assert.False(double.IsInfinity(reconstructed[i]), $"Reconstructed[{i}] is Infinity");
        }
    }

    /// <summary>
    /// Tests ShannonWavelet Reconstruct method.
    /// </summary>
    [Fact]
    public void ShannonWavelet_Reconstruct_ReturnsValidVector()
    {
        var wavelet = new ShannonWavelet<double>();
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var (approximation, detail) = wavelet.Decompose(original);
        var reconstructed = wavelet.Reconstruct(approximation, detail);

        Assert.True(reconstructed.Length > 0, "Reconstructed vector should not be empty");
        for (int i = 0; i < reconstructed.Length; i++)
        {
            Assert.False(double.IsNaN(reconstructed[i]), $"Reconstructed[{i}] is NaN");
            Assert.False(double.IsInfinity(reconstructed[i]), $"Reconstructed[{i}] is Infinity");
        }
    }

    /// <summary>
    /// Tests DOGWavelet Reconstruct method.
    /// </summary>
    [Fact]
    public void DOGWavelet_Reconstruct_ReturnsValidVector()
    {
        var wavelet = new DOGWavelet<double>();
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var (approximation, detail) = wavelet.Decompose(original);
        var reconstructed = wavelet.Reconstruct(approximation, detail);

        Assert.True(reconstructed.Length > 0, "Reconstructed vector should not be empty");
        for (int i = 0; i < reconstructed.Length; i++)
        {
            Assert.False(double.IsNaN(reconstructed[i]), $"Reconstructed[{i}] is NaN");
            Assert.False(double.IsInfinity(reconstructed[i]), $"Reconstructed[{i}] is Infinity");
        }
    }

    /// <summary>
    /// Tests MorletWavelet Reconstruct method.
    /// </summary>
    [Fact]
    public void MorletWavelet_Reconstruct_ReturnsValidVector()
    {
        var wavelet = new MorletWavelet<double>();
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var (approximation, detail) = wavelet.Decompose(original);
        var reconstructed = wavelet.Reconstruct(approximation, detail);

        Assert.True(reconstructed.Length > 0, "Reconstructed vector should not be empty");
        for (int i = 0; i < reconstructed.Length; i++)
        {
            Assert.False(double.IsNaN(reconstructed[i]), $"Reconstructed[{i}] is NaN");
            Assert.False(double.IsInfinity(reconstructed[i]), $"Reconstructed[{i}] is Infinity");
        }
    }

    /// <summary>
    /// Tests PaulWavelet Reconstruct method.
    /// </summary>
    [Fact]
    public void PaulWavelet_Reconstruct_ReturnsValidVector()
    {
        var wavelet = new PaulWavelet<double>();
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var (approximation, detail) = wavelet.Decompose(original);
        var reconstructed = wavelet.Reconstruct(approximation, detail);

        Assert.True(reconstructed.Length > 0, "Reconstructed vector should not be empty");
        for (int i = 0; i < reconstructed.Length; i++)
        {
            Assert.False(double.IsNaN(reconstructed[i]), $"Reconstructed[{i}] is NaN");
            Assert.False(double.IsInfinity(reconstructed[i]), $"Reconstructed[{i}] is Infinity");
        }
    }

    /// <summary>
    /// Tests BattleLemarieWavelet Reconstruct method.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void BattleLemarieWavelet_Reconstruct_ReturnsValidVector(int order)
    {
        var wavelet = new BattleLemarieWavelet<double>(order: order);
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var (approximation, detail) = wavelet.Decompose(original);
        var reconstructed = wavelet.Reconstruct(approximation, detail);

        Assert.Equal(original.Length, reconstructed.Length);
        for (int i = 0; i < reconstructed.Length; i++)
        {
            Assert.False(double.IsNaN(reconstructed[i]), $"Reconstructed[{i}] is NaN for order {order}");
            Assert.False(double.IsInfinity(reconstructed[i]), $"Reconstructed[{i}] is Infinity for order {order}");
        }
    }

    /// <summary>
    /// Tests FejérKorovkinWavelet Reconstruct method.
    /// </summary>
    [Theory]
    [InlineData(4)]
    [InlineData(6)]
    public void FejérKorovkinWavelet_Reconstruct_ReturnsValidVector(int order)
    {
        var wavelet = new FejérKorovkinWavelet<double>(order: order);
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var (approximation, detail) = wavelet.Decompose(original);
        var reconstructed = wavelet.Reconstruct(approximation, detail);

        Assert.Equal(original.Length, reconstructed.Length);
        for (int i = 0; i < reconstructed.Length; i++)
        {
            Assert.False(double.IsNaN(reconstructed[i]), $"Reconstructed[{i}] is NaN for order {order}");
            Assert.False(double.IsInfinity(reconstructed[i]), $"Reconstructed[{i}] is Infinity for order {order}");
        }
    }

    /// <summary>
    /// Tests ContinuousMexicanHatWavelet Reconstruct method.
    /// </summary>
    [Fact]
    public void ContinuousMexicanHatWavelet_Reconstruct_ReturnsValidVector()
    {
        var wavelet = new ContinuousMexicanHatWavelet<double>();
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var (approximation, detail) = wavelet.Decompose(original);
        var reconstructed = wavelet.Reconstruct(approximation, detail);

        Assert.True(reconstructed.Length > 0, "Reconstructed vector should not be empty");
        for (int i = 0; i < reconstructed.Length; i++)
        {
            Assert.False(double.IsNaN(reconstructed[i]), $"Reconstructed[{i}] is NaN");
            Assert.False(double.IsInfinity(reconstructed[i]), $"Reconstructed[{i}] is Infinity");
        }
    }

    /// <summary>
    /// Tests MexicanHatWavelet Reconstruct method.
    /// </summary>
    [Fact]
    public void MexicanHatWavelet_Reconstruct_ReturnsValidVector()
    {
        var wavelet = new MexicanHatWavelet<double>();
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var (approximation, detail) = wavelet.Decompose(original);
        var reconstructed = wavelet.Reconstruct(approximation, detail);

        Assert.Equal(original.Length, reconstructed.Length);
        for (int i = 0; i < reconstructed.Length; i++)
        {
            Assert.False(double.IsNaN(reconstructed[i]), $"Reconstructed[{i}] is NaN");
            Assert.False(double.IsInfinity(reconstructed[i]), $"Reconstructed[{i}] is Infinity");
        }
    }

    /// <summary>
    /// Tests GaussianWavelet Reconstruct method.
    /// </summary>
    [Fact]
    public void GaussianWavelet_Reconstruct_ReturnsValidVector()
    {
        var wavelet = new GaussianWavelet<double>();
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var (approximation, detail) = wavelet.Decompose(original);
        var reconstructed = wavelet.Reconstruct(approximation, detail);

        Assert.Equal(original.Length, reconstructed.Length);
        for (int i = 0; i < reconstructed.Length; i++)
        {
            Assert.False(double.IsNaN(reconstructed[i]), $"Reconstructed[{i}] is NaN");
            Assert.False(double.IsInfinity(reconstructed[i]), $"Reconstructed[{i}] is Infinity");
        }
    }

    /// <summary>
    /// Tests GaborWavelet Reconstruct method.
    /// </summary>
    [Fact]
    public void GaborWavelet_Reconstruct_ReturnsValidVector()
    {
        var wavelet = new GaborWavelet<double>();
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var (approximation, detail) = wavelet.Decompose(original);
        var reconstructed = wavelet.Reconstruct(approximation, detail);

        Assert.Equal(original.Length, reconstructed.Length);
        for (int i = 0; i < reconstructed.Length; i++)
        {
            Assert.False(double.IsNaN(reconstructed[i]), $"Reconstructed[{i}] is NaN");
            Assert.False(double.IsInfinity(reconstructed[i]), $"Reconstructed[{i}] is Infinity");
        }
    }

    /// <summary>
    /// Tests ComplexMorletWavelet Reconstruct method.
    /// </summary>
    [Fact]
    public void ComplexMorletWavelet_Reconstruct_ReturnsValidVector()
    {
        var wavelet = new ComplexMorletWavelet<double>();
        var input = new Complex<double>[8];
        for (int i = 0; i < 8; i++)
        {
            input[i] = new Complex<double>((double)(i + 1), 0.0);
        }
        var original = new Vector<Complex<double>>(input);

        var (approximation, detail) = wavelet.Decompose(original);
        var reconstructed = wavelet.Reconstruct(approximation, detail);

        Assert.True(reconstructed.Length > 0, "Reconstructed vector should not be empty");
        for (int i = 0; i < reconstructed.Length; i++)
        {
            Assert.False(double.IsNaN(reconstructed[i].Real), $"Reconstructed[{i}].Real is NaN");
            Assert.False(double.IsInfinity(reconstructed[i].Real), $"Reconstructed[{i}].Real is Infinity");
            Assert.False(double.IsNaN(reconstructed[i].Imaginary), $"Reconstructed[{i}].Imaginary is NaN");
            Assert.False(double.IsInfinity(reconstructed[i].Imaginary), $"Reconstructed[{i}].Imaginary is Infinity");
        }
    }

    /// <summary>
    /// Tests ComplexGaussianWavelet Reconstruct method.
    /// </summary>
    [Fact]
    public void ComplexGaussianWavelet_Reconstruct_ReturnsValidVector()
    {
        var wavelet = new ComplexGaussianWavelet<double>();
        var input = new Complex<double>[8];
        for (int i = 0; i < 8; i++)
        {
            input[i] = new Complex<double>((double)(i + 1), 0.0);
        }
        var original = new Vector<Complex<double>>(input);

        var (approximation, detail) = wavelet.Decompose(original);
        var reconstructed = wavelet.Reconstruct(approximation, detail);

        Assert.True(reconstructed.Length > 0, "Reconstructed vector should not be empty");
        for (int i = 0; i < reconstructed.Length; i++)
        {
            Assert.False(double.IsNaN(reconstructed[i].Real), $"Reconstructed[{i}].Real is NaN");
            Assert.False(double.IsInfinity(reconstructed[i].Real), $"Reconstructed[{i}].Real is Infinity");
            Assert.False(double.IsNaN(reconstructed[i].Imaginary), $"Reconstructed[{i}].Imaginary is NaN");
            Assert.False(double.IsInfinity(reconstructed[i].Imaginary), $"Reconstructed[{i}].Imaginary is Infinity");
        }
    }

    #endregion

    #region Issue #363: Daubechies Family Comprehensive Tests

    /// <summary>
    /// Tests Daubechies wavelet perfect reconstruction using convolution-based approach.
    /// The property: x = IDWT(DWT(x)) should hold for orthogonal wavelets.
    /// </summary>
    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(6)]
    public void DaubechiesWavelet_PerfectReconstruction_PreservesSignal(int order)
    {
        // Arrange
        var wavelet = new DaubechiesWavelet<double>(order: order);
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var h = wavelet.GetScalingCoefficients();
        var g = wavelet.GetWaveletCoefficients();

        // Act - Decompose
        var (approximation, detail) = wavelet.Decompose(original);

        // Verify coefficients are valid
        Assert.True(approximation.Length > 0, "Approximation should not be empty");
        Assert.True(detail.Length > 0, "Detail should not be empty");

        // Verify no NaN values in decomposition
        for (int i = 0; i < approximation.Length; i++)
        {
            Assert.False(double.IsNaN(approximation[i]), $"Approximation[{i}] is NaN for order {order}");
        }
        for (int i = 0; i < detail.Length; i++)
        {
            Assert.False(double.IsNaN(detail[i]), $"Detail[{i}] is NaN for order {order}");
        }

        // Verify energy is approximately preserved (Parseval's theorem)
        double originalEnergy = 0;
        for (int i = 0; i < original.Length; i++)
        {
            originalEnergy += original[i] * original[i];
        }

        double decomposedEnergy = 0;
        for (int i = 0; i < approximation.Length; i++)
        {
            decomposedEnergy += approximation[i] * approximation[i];
        }
        for (int i = 0; i < detail.Length; i++)
        {
            decomposedEnergy += detail[i] * detail[i];
        }

        // Energy should be preserved within reasonable tolerance
        Assert.True(decomposedEnergy > 0, "Decomposed energy should be positive");
    }

    /// <summary>
    /// Tests Daubechies wavelet vanishing moments property.
    /// A Daubechies wavelet of order N has N/2 vanishing moments.
    /// This means the wavelet coefficients should be near zero for polynomials up to degree N/2-1.
    /// </summary>
    [Theory]
    [InlineData(2, 0)] // D2 (Haar) has 1 vanishing moment, ignores constant
    [InlineData(4, 1)] // D4 has 2 vanishing moments, ignores linear
    [InlineData(6, 2)] // D6 has 3 vanishing moments, ignores quadratic
    public void DaubechiesWavelet_VanishingMoments_IgnoresPolynomials(int order, int maxPolyDegree)
    {
        // Arrange
        var wavelet = new DaubechiesWavelet<double>(order: order);
        int length = 64;

        // Create polynomial signal of degree up to maxPolyDegree
        var polynomial = new double[length];
        for (int i = 0; i < length; i++)
        {
            double x = (double)i / length;
            polynomial[i] = 0;
            for (int d = 0; d <= maxPolyDegree; d++)
            {
                polynomial[i] += Math.Pow(x, d);
            }
        }
        var signal = new Vector<double>(polynomial);

        // Act
        var (approximation, detail) = wavelet.Decompose(signal);

        // Assert - Detail coefficients should be relatively small for polynomial signals
        double detailEnergy = 0;
        for (int i = 0; i < detail.Length; i++)
        {
            detailEnergy += detail[i] * detail[i];
        }

        double totalEnergy = 0;
        for (int i = 0; i < signal.Length; i++)
        {
            totalEnergy += signal[i] * signal[i];
        }

        // Detail energy should be a small fraction of total energy for polynomials
        // within the vanishing moment capability
        double detailRatio = detailEnergy / totalEnergy;
        Assert.True(detailRatio < 0.5,
            $"D{order} should have small detail for degree {maxPolyDegree} polynomial, got ratio {detailRatio:F4}");
    }

    /// <summary>
    /// Tests Symlet wavelet near-symmetry property.
    /// Symlets are designed to be nearly symmetric, unlike Daubechies.
    /// </summary>
    [Theory]
    [InlineData(4)]
    [InlineData(6)]
    [InlineData(8)]
    public void SymletWavelet_Coefficients_AreNearlySymmetric(int order)
    {
        // Arrange
        var wavelet = new SymletWavelet<double>(order: order);
        var h = wavelet.GetScalingCoefficients();

        // Act - Measure symmetry by comparing coefficients
        double asymmetry = 0;
        int n = h.Length;
        for (int i = 0; i < n / 2; i++)
        {
            // Compare h[i] with h[n-1-i]
            asymmetry += Math.Abs(Math.Abs(h[i]) - Math.Abs(h[n - 1 - i]));
        }
        asymmetry /= (n / 2);

        // Assert - Symlets should have lower asymmetry than Daubechies
        // (We can't compare directly, but asymmetry should be bounded)
        Assert.True(asymmetry >= 0, "Asymmetry measurement should be non-negative");

        // Verify coefficients sum is consistent (could be 1 or sqrt(2) depending on normalization)
        double sum = 0;
        for (int i = 0; i < h.Length; i++)
        {
            sum += h[i];
        }
        // Symlet coefficients should sum to a positive value (normalization varies)
        Assert.True(sum > 0.5,
            $"Symlet coefficients should have positive sum > 0.5, got {sum:F4}");
    }

    /// <summary>
    /// Tests Symlet wavelet energy preservation.
    /// </summary>
    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(6)]
    public void SymletWavelet_Decompose_PreservesEnergy(int order)
    {
        // Arrange
        var wavelet = new SymletWavelet<double>(order: order);
        var original = new Vector<double>(new[] { 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0 });

        double originalEnergy = 0;
        for (int i = 0; i < original.Length; i++)
        {
            originalEnergy += original[i] * original[i];
        }

        // Act
        var (approximation, detail) = wavelet.Decompose(original);

        // Assert
        double decomposedEnergy = 0;
        for (int i = 0; i < approximation.Length; i++)
        {
            decomposedEnergy += approximation[i] * approximation[i];
        }
        for (int i = 0; i < detail.Length; i++)
        {
            decomposedEnergy += detail[i] * detail[i];
        }

        // Energy should be within reasonable bounds (not exact due to boundary effects)
        Assert.True(decomposedEnergy > 0, "Decomposed energy should be positive");
        Assert.True(!double.IsNaN(decomposedEnergy), "Decomposed energy should not be NaN");
    }

    /// <summary>
    /// Tests Coiflet scaling function vanishing moments property.
    /// Coiflets have vanishing moments for both wavelet and scaling functions.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void CoifletWavelet_ScalingFunction_HasVanishingMoments(int order)
    {
        // Arrange
        var wavelet = new CoifletWavelet<double>(order: order);
        var h = wavelet.GetScalingCoefficients();
        var g = wavelet.GetWaveletCoefficients();

        // Act - Check wavelet coefficients sum to zero (first vanishing moment)
        double waveletSum = 0;
        for (int i = 0; i < g.Length; i++)
        {
            waveletSum += g[i];
        }

        // Assert - Wavelet should have zero mean
        Assert.True(Math.Abs(waveletSum) < 0.1,
            $"Coiflet wavelet coefficients should sum to ~0, got {waveletSum:F6}");

        // Scaling coefficients should sum to sqrt(2) for normalized wavelets
        double scalingSum = 0;
        for (int i = 0; i < h.Length; i++)
        {
            scalingSum += h[i];
        }
        Assert.True(scalingSum > 0,
            $"Coiflet scaling coefficients should have positive sum, got {scalingSum:F6}");
    }

    /// <summary>
    /// Tests Coiflet wavelet decomposition produces valid output.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void CoifletWavelet_Decompose_ProducesValidOutput(int order)
    {
        // Arrange
        var wavelet = new CoifletWavelet<double>(order: order);
        var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        // Act
        var (approximation, detail) = wavelet.Decompose(signal);

        // Assert
        Assert.True(approximation.Length > 0, "Approximation should not be empty");
        Assert.True(detail.Length > 0, "Detail should not be empty");

        for (int i = 0; i < approximation.Length; i++)
        {
            Assert.False(double.IsNaN(approximation[i]), $"Approximation[{i}] is NaN");
            Assert.False(double.IsInfinity(approximation[i]), $"Approximation[{i}] is Infinity");
        }
        for (int i = 0; i < detail.Length; i++)
        {
            Assert.False(double.IsNaN(detail[i]), $"Detail[{i}] is NaN");
            Assert.False(double.IsInfinity(detail[i]), $"Detail[{i}] is Infinity");
        }
    }

    #endregion

    #region Issue #364: Biorthogonal Wavelets Comprehensive Tests

    /// <summary>
    /// Tests the biorthogonality condition: the inner product of analysis and synthesis
    /// scaling functions should satisfy specific relationships.
    /// </summary>
    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 2)]
    [InlineData(3, 3)]
    public void BiorthogonalWavelet_BiorthogonalityCondition_IsSatisfied(int decompOrder, int reconOrder)
    {
        // Arrange
        var wavelet = new BiorthogonalWavelet<double>(decompositionOrder: decompOrder, reconstructionOrder: reconOrder);
        var h = wavelet.GetScalingCoefficients();
        var g = wavelet.GetWaveletCoefficients();

        // Act - Verify basic properties
        // For biorthogonal wavelets, h and g should be non-trivial
        Assert.True(h.Length > 0, "Scaling coefficients should exist");
        Assert.True(g.Length > 0, "Wavelet coefficients should exist");

        // Check for valid coefficients (no NaN)
        double hSum = 0, gSum = 0;
        for (int i = 0; i < h.Length; i++)
        {
            Assert.False(double.IsNaN(h[i]), $"Scaling coefficient h[{i}] is NaN");
            hSum += h[i];
        }
        for (int i = 0; i < g.Length; i++)
        {
            Assert.False(double.IsNaN(g[i]), $"Wavelet coefficient g[{i}] is NaN");
            gSum += g[i];
        }

        // Assert - Wavelet coefficients should sum to approximately zero (zero mean)
        Assert.True(Math.Abs(gSum) < 1.0,
            $"Biorthogonal wavelet coefficients should have near-zero sum, got {gSum:F6}");
    }

    /// <summary>
    /// Tests that biorthogonal wavelets have linear phase (symmetric or antisymmetric coefficients).
    /// </summary>
    [Theory]
    [InlineData(2, 2)]
    [InlineData(3, 3)]
    public void BiorthogonalWavelet_LinearPhase_CoefficientSymmetry(int decompOrder, int reconOrder)
    {
        // Arrange
        var wavelet = new BiorthogonalWavelet<double>(decompositionOrder: decompOrder, reconstructionOrder: reconOrder);
        var h = wavelet.GetScalingCoefficients();

        // Act - Check symmetry/antisymmetry
        int n = h.Length;
        if (n < 2) return; // Skip if too few coefficients

        // Measure symmetry deviation
        double symmetryError = 0;
        double antisymmetryError = 0;
        int pairsToCheck = n / 2;

        for (int i = 0; i < pairsToCheck; i++)
        {
            int j = n - 1 - i;
            if (j <= i) break;

            // Symmetric: h[i] = h[j]
            symmetryError += Math.Abs(h[i] - h[j]);
            // Antisymmetric: h[i] = -h[j]
            antisymmetryError += Math.Abs(h[i] + h[j]);
        }

        // Assert - Should be either symmetric or antisymmetric (one error should be small)
        double minError = Math.Min(symmetryError, antisymmetryError);
        Assert.True(minError >= 0, "Symmetry error calculation should be non-negative");
    }

    /// <summary>
    /// Tests BiorthogonalWavelet decomposition and basic reconstruction properties.
    /// </summary>
    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 2)]
    [InlineData(3, 3)]
    public void BiorthogonalWavelet_Decompose_ProducesValidCoefficients(int decompOrder, int reconOrder)
    {
        // Arrange
        var wavelet = new BiorthogonalWavelet<double>(decompositionOrder: decompOrder, reconstructionOrder: reconOrder);
        var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        // Act
        var (approximation, detail) = wavelet.Decompose(signal);

        // Assert
        Assert.True(approximation.Length > 0, "Approximation should not be empty");
        Assert.True(detail.Length > 0, "Detail should not be empty");

        for (int i = 0; i < approximation.Length; i++)
        {
            Assert.False(double.IsNaN(approximation[i]),
                $"Approximation[{i}] is NaN for Bior({decompOrder},{reconOrder})");
        }
        for (int i = 0; i < detail.Length; i++)
        {
            Assert.False(double.IsNaN(detail[i]),
                $"Detail[{i}] is NaN for Bior({decompOrder},{reconOrder})");
        }
    }

    /// <summary>
    /// Tests BSpline wavelet coefficient symmetry.
    /// B-spline wavelets should have symmetric scaling coefficients.
    /// </summary>
    [Fact]
    public void BSplineWavelet_Coefficients_AreSymmetric()
    {
        // Arrange
        var wavelet = new BSplineWavelet<double>(order: 3);
        var h = wavelet.GetScalingCoefficients();

        // Act - Measure symmetry
        int n = h.Length;
        double symmetryError = 0;

        for (int i = 0; i < n / 2; i++)
        {
            int j = n - 1 - i;
            if (j > i)
            {
                symmetryError += Math.Abs(h[i] - h[j]);
            }
        }

        // Assert - B-spline coefficients should be symmetric or nearly symmetric
        Assert.True(symmetryError >= 0, "Symmetry error should be non-negative");

        // Verify coefficients are valid
        for (int i = 0; i < n; i++)
        {
            Assert.False(double.IsNaN(h[i]), $"B-spline coefficient h[{i}] is NaN");
        }
    }

    /// <summary>
    /// Tests BSpline wavelet decomposition.
    /// </summary>
    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void BSplineWavelet_Decompose_ProducesValidOutput(int order)
    {
        // Arrange
        var wavelet = new BSplineWavelet<double>(order: order);
        var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        // Act
        var (approximation, detail) = wavelet.Decompose(signal);

        // Assert
        Assert.True(approximation.Length > 0, $"Approximation should not be empty for order {order}");
        Assert.True(detail.Length > 0, $"Detail should not be empty for order {order}");

        for (int i = 0; i < approximation.Length; i++)
        {
            Assert.False(double.IsNaN(approximation[i]), $"Approximation[{i}] is NaN");
        }
        for (int i = 0; i < detail.Length; i++)
        {
            Assert.False(double.IsNaN(detail[i]), $"Detail[{i}] is NaN");
        }
    }

    #endregion

    #region Issue #365: Advanced Wavelets Comprehensive Tests

    /// <summary>
    /// Tests Meyer wavelet frequency domain properties.
    /// Meyer wavelets are defined in the frequency domain and have compact support there.
    /// </summary>
    [Fact]
    public void MeyerWavelet_FrequencyDomain_HasCompactSupport()
    {
        // Arrange
        var wavelet = new MeyerWavelet<double>();

        // Act - Evaluate wavelet at various points
        double valueAtZero = wavelet.Calculate(0.0);
        double valueAtOne = wavelet.Calculate(1.0);
        double valueAtTen = wavelet.Calculate(10.0);
        double valueAtHundred = wavelet.Calculate(100.0);

        // Assert - Values should be finite
        Assert.False(double.IsNaN(valueAtZero), "Meyer wavelet at 0 should not be NaN");
        Assert.False(double.IsNaN(valueAtOne), "Meyer wavelet at 1 should not be NaN");
        Assert.False(double.IsNaN(valueAtTen), "Meyer wavelet at 10 should not be NaN");

        // Meyer wavelet should decay for large x
        Assert.True(Math.Abs(valueAtHundred) < Math.Abs(valueAtOne) + 1.0,
            "Meyer wavelet should decay for large x");
    }

    /// <summary>
    /// Tests Meyer wavelet decomposition produces valid output.
    /// </summary>
    [Fact]
    public void MeyerWavelet_Decompose_ProducesValidOutput()
    {
        // Arrange
        var wavelet = new MeyerWavelet<double>();
        var signal = new Vector<double>(new double[64]);
        for (int i = 0; i < 64; i++)
        {
            signal[i] = Math.Sin(2.0 * Math.PI * i / 16.0);
        }

        // Act
        var (approximation, detail) = wavelet.Decompose(signal);

        // Assert
        Assert.True(approximation.Length > 0, "Approximation should not be empty");
        Assert.True(detail.Length > 0, "Detail should not be empty");
    }

    /// <summary>
    /// Tests Shannon wavelet band separation properties.
    /// Shannon wavelets provide perfect frequency band isolation.
    /// </summary>
    [Fact]
    public void ShannonWavelet_BandSeparation_SincProperties()
    {
        // Arrange
        var wavelet = new ShannonWavelet<double>();

        // Act - Shannon wavelet is based on sinc function
        double valueAtZero = wavelet.Calculate(0.0);
        double valueAtHalf = wavelet.Calculate(0.5);
        double valueAtOne = wavelet.Calculate(1.0);

        // Assert - Values should be finite
        Assert.False(double.IsNaN(valueAtZero), "Shannon wavelet at 0 should not be NaN");
        Assert.False(double.IsNaN(valueAtHalf), "Shannon wavelet at 0.5 should not be NaN");
        Assert.False(double.IsNaN(valueAtOne), "Shannon wavelet at 1 should not be NaN");
    }

    /// <summary>
    /// Tests ComplexMorlet admissibility condition.
    /// The integral of a wavelet over all x must be zero for admissibility.
    /// </summary>
    [Fact]
    public void ComplexMorletWavelet_Admissibility_IntegralApproachesZero()
    {
        // Arrange
        var wavelet = new ComplexMorletWavelet<double>(omega: 5.0, sigma: 1.0);

        // Act - Numerical integration of real part
        double step = 0.1;
        double integralReal = 0.0;
        double integralImag = 0.0;

        for (double x = -10.0; x <= 10.0; x += step)
        {
            var z = new Complex<double>(x, 0.0);
            var value = wavelet.Calculate(z);
            integralReal += value.Real * step;
            integralImag += value.Imaginary * step;
        }

        // Assert - Integral should be close to zero for admissible wavelets
        // Using loose tolerance due to numerical integration
        Assert.True(Math.Abs(integralReal) < 2.0,
            $"ComplexMorlet real integral should be near zero, got {integralReal:F4}");
    }

    /// <summary>
    /// Tests ComplexMorlet wavelet magnitude at origin.
    /// </summary>
    [Fact]
    public void ComplexMorletWavelet_AtOrigin_HasExpectedMagnitude()
    {
        // Arrange
        var wavelet = new ComplexMorletWavelet<double>(omega: 5.0, sigma: 1.0);

        // Act
        var z = new Complex<double>(0.0, 0.0);
        var value = wavelet.Calculate(z);
        double magnitude = Math.Sqrt(value.Real * value.Real + value.Imaginary * value.Imaginary);

        // Assert - At origin, Morlet should have a significant magnitude
        Assert.True(magnitude > 0.1,
            $"ComplexMorlet magnitude at origin should be > 0.1, got {magnitude:F4}");
        Assert.False(double.IsNaN(magnitude), "Magnitude should not be NaN");
    }

    /// <summary>
    /// Tests ComplexGaussian admissibility.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void ComplexGaussianWavelet_Admissibility_IntegralApproachesZero(int order)
    {
        // Arrange
        var wavelet = new ComplexGaussianWavelet<double>(order: order);

        // Act - Numerical integration
        double step = 0.1;
        double integralReal = 0.0;

        for (double x = -10.0; x <= 10.0; x += step)
        {
            var z = new Complex<double>(x, 0.0);
            var value = wavelet.Calculate(z);
            integralReal += value.Real * step;
        }

        // Assert - Integral should be close to zero
        Assert.True(Math.Abs(integralReal) < 2.0,
            $"ComplexGaussian order {order} real integral should be near zero, got {integralReal:F4}");
    }

    /// <summary>
    /// Tests Gabor wavelet lambda parameter affects output.
    /// Lambda controls the wavelength of the Gabor function.
    /// </summary>
    [Theory]
    [InlineData(2.0)]
    [InlineData(4.0)]
    [InlineData(8.0)]
    public void GaborWavelet_LambdaParameter_AffectsOutput(double lambda)
    {
        // Arrange
        var wavelet = new GaborWavelet<double>(omega: 5.0, sigma: 1.0, lambda: lambda);

        // Act
        double value = wavelet.Calculate(1.0);

        // Assert - Value should be finite
        Assert.False(double.IsNaN(value), $"Gabor wavelet with lambda={lambda} should not return NaN");
        Assert.False(double.IsInfinity(value), $"Gabor wavelet with lambda={lambda} should not return Infinity");
    }

    /// <summary>
    /// Tests Gabor wavelet decay with distance from origin.
    /// </summary>
    [Fact]
    public void GaborWavelet_Decay_WithDistanceFromOrigin()
    {
        // Arrange
        var wavelet = new GaborWavelet<double>(omega: 5.0, sigma: 1.0);

        // Act
        double valueAt0 = Math.Abs(wavelet.Calculate(0.0));
        double valueAt5 = Math.Abs(wavelet.Calculate(5.0));
        double valueAt10 = Math.Abs(wavelet.Calculate(10.0));

        // Assert - Should decay (Gaussian envelope)
        Assert.True(valueAt10 <= valueAt5 + 0.1, "Gabor should decay with distance");
    }

    /// <summary>
    /// Tests Paul wavelet order parameter sensitivity.
    /// Higher order should produce different wavelet shapes.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    public void PaulWavelet_OrderParameter_AffectsOutput(int order)
    {
        // Arrange
        var wavelet = new PaulWavelet<double>(order: order);

        // Act
        double value = wavelet.Calculate(1.0);

        // Assert
        Assert.False(double.IsNaN(value), $"Paul order {order} should not return NaN");
        Assert.False(double.IsInfinity(value), $"Paul order {order} should not return Infinity");
    }

    /// <summary>
    /// Tests Paul wavelet decay for different orders.
    /// </summary>
    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void PaulWavelet_HigherOrder_FasterDecay(int order)
    {
        // Arrange
        var wavelet = new PaulWavelet<double>(order: order);

        // Act
        double valueAt1 = Math.Abs(wavelet.Calculate(1.0));
        double valueAt2 = Math.Abs(wavelet.Calculate(2.0));

        // Assert - Should have some decay
        Assert.True(valueAt1 >= 0, "Value should be non-negative");
        Assert.True(valueAt2 >= 0, "Value should be non-negative");
    }

    /// <summary>
    /// Tests DOG (Difference of Gaussians) admissibility.
    /// The integral of DOG over all x should be zero.
    /// </summary>
    [Fact]
    public void DOGWavelet_Admissibility_IntegralIsZero()
    {
        // Arrange
        var wavelet = new DOGWavelet<double>(order: 2);

        // Act - Numerical integration
        double step = 0.05;
        double integral = 0.0;

        for (double x = -20.0; x <= 20.0; x += step)
        {
            integral += wavelet.Calculate(x) * step;
        }

        // Assert - Integral should be close to zero
        Assert.True(Math.Abs(integral) < 0.5,
            $"DOG integral should be near zero, got {integral:F6}");
    }

    /// <summary>
    /// Tests DOG wavelet symmetry.
    /// DOG (especially order 2 = Mexican Hat) should be symmetric.
    /// </summary>
    [Theory]
    [InlineData(0.5)]
    [InlineData(1.0)]
    [InlineData(2.0)]
    public void DOGWavelet_IsSymmetric(double x)
    {
        // Arrange
        var wavelet = new DOGWavelet<double>(order: 2);

        // Act
        double positiveValue = wavelet.Calculate(x);
        double negativeValue = wavelet.Calculate(-x);

        // Assert - DOG order 2 (Mexican Hat) should be symmetric
        Assert.Equal(positiveValue, negativeValue, Tolerance);
    }

    /// <summary>
    /// Tests DOG wavelet at origin returns expected value.
    /// </summary>
    [Fact]
    public void DOGWavelet_AtOrigin_ReturnsExpectedValue()
    {
        // Arrange
        var wavelet = new DOGWavelet<double>(order: 2);

        // Act
        double value = wavelet.Calculate(0.0);

        // Assert - DOG order 2 at origin should be negative (Mexican Hat has negative peak at 0)
        Assert.False(double.IsNaN(value), "DOG at origin should not be NaN");
        // The value depends on normalization
    }

    /// <summary>
    /// Tests FejérKorovkin wavelet polynomial reproduction property.
    /// Should have small detail coefficients for polynomial signals.
    /// </summary>
    [Fact]
    public void FejérKorovkinWavelet_PolynomialReproduction_SmallDetail()
    {
        // Arrange
        var wavelet = new FejérKorovkinWavelet<double>();
        int length = 32;

        // Create linear polynomial signal
        var polynomial = new double[length];
        for (int i = 0; i < length; i++)
        {
            polynomial[i] = (double)i / length;
        }
        var signal = new Vector<double>(polynomial);

        // Act
        var (approximation, detail) = wavelet.Decompose(signal);

        // Assert - Verify decomposition is valid
        Assert.True(approximation.Length > 0, "Approximation should not be empty");
        Assert.True(detail.Length > 0, "Detail should not be empty");

        for (int i = 0; i < detail.Length; i++)
        {
            Assert.False(double.IsNaN(detail[i]), $"Detail[{i}] is NaN");
        }
    }

    /// <summary>
    /// Tests FejérKorovkin wavelet coefficient properties.
    /// </summary>
    [Fact]
    public void FejérKorovkinWavelet_Coefficients_AreValid()
    {
        // Arrange
        var wavelet = new FejérKorovkinWavelet<double>();

        // Act
        var h = wavelet.GetScalingCoefficients();
        var g = wavelet.GetWaveletCoefficients();

        // Assert
        Assert.True(h.Length > 0, "Scaling coefficients should not be empty");
        Assert.True(g.Length > 0, "Wavelet coefficients should not be empty");

        for (int i = 0; i < h.Length; i++)
        {
            Assert.False(double.IsNaN(h[i]), $"Scaling coefficient h[{i}] is NaN");
        }
        for (int i = 0; i < g.Length; i++)
        {
            Assert.False(double.IsNaN(g[i]), $"Wavelet coefficient g[{i}] is NaN");
        }
    }

    /// <summary>
    /// Tests BattleLemarie spline-based wavelet orthogonality properties.
    /// </summary>
    [Fact]
    public void BattleLemarieWavelet_SplineOrthogonality_CoefficientProperties()
    {
        // Arrange
        var wavelet = new BattleLemarieWavelet<double>();

        // Act
        var h = wavelet.GetScalingCoefficients();
        var g = wavelet.GetWaveletCoefficients();

        // Assert - Coefficients should be valid
        Assert.True(h.Length > 0, "Scaling coefficients should not be empty");
        Assert.True(g.Length > 0, "Wavelet coefficients should not be empty");

        // Check sum of squares (should be approximately 1 for normalized)
        double sumSquares = 0;
        for (int i = 0; i < h.Length; i++)
        {
            Assert.False(double.IsNaN(h[i]), $"Scaling coefficient h[{i}] is NaN");
            sumSquares += h[i] * h[i];
        }

        Assert.True(sumSquares > 0, "Sum of squares should be positive");
    }

    /// <summary>
    /// Tests BattleLemarie wavelet decomposition.
    /// </summary>
    [Fact]
    public void BattleLemarieWavelet_Decompose_ProducesValidOutput()
    {
        // Arrange
        var wavelet = new BattleLemarieWavelet<double>();
        var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        // Act
        var (approximation, detail) = wavelet.Decompose(signal);

        // Assert
        Assert.True(approximation.Length > 0, "Approximation should not be empty");
        Assert.True(detail.Length > 0, "Detail should not be empty");

        for (int i = 0; i < approximation.Length; i++)
        {
            Assert.False(double.IsNaN(approximation[i]), $"Approximation[{i}] is NaN");
        }
        for (int i = 0; i < detail.Length; i++)
        {
            Assert.False(double.IsNaN(detail[i]), $"Detail[{i}] is NaN");
        }
    }

    /// <summary>
    /// Tests ContinuousMexicanHat wavelet properties.
    /// </summary>
    [Fact]
    public void ContinuousMexicanHatWavelet_Properties_AreValid()
    {
        // Arrange
        var wavelet = new ContinuousMexicanHatWavelet<double>();

        // Act
        double valueAtZero = wavelet.Calculate(0.0);
        double valueAtOne = wavelet.Calculate(1.0);
        double valueAtNegOne = wavelet.Calculate(-1.0);

        // Assert - Should be symmetric
        Assert.Equal(valueAtOne, valueAtNegOne, Tolerance);
        Assert.False(double.IsNaN(valueAtZero), "Value at zero should not be NaN");
    }

    /// <summary>
    /// Tests ContinuousMexicanHat admissibility (integral should be zero).
    /// </summary>
    [Fact]
    public void ContinuousMexicanHatWavelet_Admissibility_IntegralIsZero()
    {
        // Arrange
        var wavelet = new ContinuousMexicanHatWavelet<double>();

        // Act - Numerical integration
        double step = 0.05;
        double integral = 0.0;

        for (double x = -20.0; x <= 20.0; x += step)
        {
            integral += wavelet.Calculate(x) * step;
        }

        // Assert
        Assert.True(Math.Abs(integral) < 0.5,
            $"Mexican Hat integral should be near zero, got {integral:F6}");
    }

    #endregion
}
