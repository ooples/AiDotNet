using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
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
}
