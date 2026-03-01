using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.WaveletFunctions;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.WaveletFunctions;

/// <summary>
/// Integration tests for advanced wavelet function classes not covered by basic tests.
/// Covers BSpline, BattleLemarie, ComplexGaussian, ComplexMorlet, FejérKorovkin,
/// Gabor, ReverseBiorthogonal, and additional mathematical property tests.
/// </summary>
public class AdvancedWaveletFunctionsTests
{
    private const double Tolerance = 1e-6;

    #region BSpline Wavelet Tests

    [Fact]
    public void BSplineWavelet_Calculate_NoNaN()
    {
        var bspline = new BSplineWavelet<double>(3);
        var values = new[] { -2.0, -1.0, 0.0, 0.5, 1.0, 2.0 };
        foreach (var x in values)
        {
            var result = bspline.Calculate(x);
            Assert.False(double.IsNaN(result));
        }
    }

    [Fact]
    public void BSplineWavelet_GetScalingCoefficients_NonEmpty()
    {
        var bspline = new BSplineWavelet<double>(3);
        var coeffs = bspline.GetScalingCoefficients();
        Assert.True(coeffs.Length > 0);
    }

    [Fact]
    public void BSplineWavelet_Decompose_EvenInput()
    {
        var bspline = new BSplineWavelet<double>(2);
        var input = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
        var (approx, detail) = bspline.Decompose(input);
        Assert.Equal(3, approx.Length);
        Assert.Equal(3, detail.Length);
    }

    #endregion

    #region BattleLemarie Wavelet Tests

    [Fact]
    public void BattleLemarieWavelet_Calculate_NoNaN()
    {
        var bl = new BattleLemarieWavelet<double>(2);
        var values = new[] { -2.0, 0.0, 1.0, 2.0 };
        foreach (var x in values)
        {
            var result = bl.Calculate(x);
            Assert.False(double.IsNaN(result));
        }
    }

    [Fact]
    public void BattleLemarieWavelet_GetScalingCoefficients_NonEmpty()
    {
        var bl = new BattleLemarieWavelet<double>(2);
        var coeffs = bl.GetScalingCoefficients();
        Assert.True(coeffs.Length > 0);
    }

    #endregion

    // ComplexGaussianWavelet and ComplexMorletWavelet use Complex<T> type
    // and extend ComplexWaveletFunctionBase<T>, tested separately if needed

    #region FejérKorovkin Wavelet Tests

    [Fact]
    public void FejérKorovkinWavelet_Calculate_NoNaN()
    {
        var fk = new FejérKorovkinWavelet<double>(4);
        var values = new[] { -2.0, 0.0, 1.0, 2.0 };
        foreach (var x in values)
        {
            var result = fk.Calculate(x);
            Assert.False(double.IsNaN(result));
        }
    }

    [Fact]
    public void FejérKorovkinWavelet_GetScalingCoefficients_NonEmpty()
    {
        var fk = new FejérKorovkinWavelet<double>(4);
        var coeffs = fk.GetScalingCoefficients();
        Assert.True(coeffs.Length > 0);
    }

    #endregion

    #region Gabor Wavelet Tests

    [Fact]
    public void GaborWavelet_Calculate_AtZero()
    {
        var gabor = new GaborWavelet<double>();
        var result = gabor.Calculate(0.0);
        Assert.False(double.IsNaN(result));
    }

    [Fact]
    public void GaborWavelet_Calculate_DecaysFromCenter()
    {
        var gabor = new GaborWavelet<double>();
        var atZero = Math.Abs(gabor.Calculate(0.0));
        var atFar = Math.Abs(gabor.Calculate(10.0));
        Assert.True(atZero >= atFar);
    }

    #endregion

    #region ReverseBiorthogonal Wavelet Tests

    [Fact]
    public void ReverseBiorthogonalWavelet_Calculate_NoNaN()
    {
        var rbio = new ReverseBiorthogonalWavelet<double>();
        var values = new[] { -2.0, 0.0, 1.0, 2.0 };
        foreach (var x in values)
        {
            var result = rbio.Calculate(x);
            Assert.False(double.IsNaN(result));
        }
    }

    [Fact]
    public void ReverseBiorthogonalWavelet_Decompose_EvenInput()
    {
        var rbio = new ReverseBiorthogonalWavelet<double>();
        var input = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 });
        var (approx, detail) = rbio.Decompose(input);
        Assert.Equal(2, approx.Length);
        Assert.Equal(2, detail.Length);
    }

    #endregion

    #region ContinuousMexicanHat Wavelet Tests

    [Fact]
    public void ContinuousMexicanHatWavelet_Calculate_PeaksAtZero()
    {
        var cmh = new ContinuousMexicanHatWavelet<double>();
        var atZero = cmh.Calculate(0.0);
        var atOne = cmh.Calculate(1.0);
        Assert.True(atZero > atOne);
    }

    [Fact]
    public void ContinuousMexicanHatWavelet_Calculate_IsEvenFunction()
    {
        var cmh = new ContinuousMexicanHatWavelet<double>();
        var pos = cmh.Calculate(1.0);
        var neg = cmh.Calculate(-1.0);
        Assert.Equal(pos, neg, Tolerance);
    }

    #endregion

    #region Haar Perfect Reconstruction Stress Test

    [Fact]
    public void HaarWavelet_DecomposeReconstruct_LargerInput_PerfectReconstruction()
    {
        var haar = new HaarWavelet<double>();
        var input = new Vector<double>(new double[] { 1, 4, -3, 0, 2, 7, -1, 5 });
        var (approx, detail) = haar.Decompose(input);
        var reconstructed = haar.Reconstruct(approx, detail);
        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(input[i], reconstructed[i], Tolerance);
        }
    }

    [Fact]
    public void HaarWavelet_Decompose_ConstantSignal_DetailIsZero()
    {
        var haar = new HaarWavelet<double>();
        var input = new Vector<double>(new double[] { 5.0, 5.0, 5.0, 5.0 });
        var (_, detail) = haar.Decompose(input);
        for (int i = 0; i < detail.Length; i++)
        {
            Assert.Equal(0.0, detail[i], Tolerance);
        }
    }

    #endregion

    #region All Wavelets GetScalingCoefficients Test

    [Fact]
    public void AllDiscreteWavelets_GetScalingCoefficients_NonEmpty()
    {
        var wavelets = new IWaveletFunction<double>[]
        {
            new HaarWavelet<double>(),
            new DaubechiesWavelet<double>(2),
            new CoifletWavelet<double>(1),
            new SymletWavelet<double>(4),
            new BiorthogonalWavelet<double>(2, 2),
            new BSplineWavelet<double>(2),
            new BattleLemarieWavelet<double>(2),
            new FejérKorovkinWavelet<double>(4),
            new ReverseBiorthogonalWavelet<double>(),
        };

        foreach (var wavelet in wavelets)
        {
            var coeffs = wavelet.GetScalingCoefficients();
            Assert.True(coeffs.Length > 0,
                $"{wavelet.GetType().Name}.GetScalingCoefficients() returned empty");
        }
    }

    [Fact]
    public void AllDiscreteWavelets_GetWaveletCoefficients_NonEmpty()
    {
        var wavelets = new IWaveletFunction<double>[]
        {
            new HaarWavelet<double>(),
            new DaubechiesWavelet<double>(2),
            new CoifletWavelet<double>(1),
            new SymletWavelet<double>(4),
            new BiorthogonalWavelet<double>(2, 2),
            new BSplineWavelet<double>(2),
            new BattleLemarieWavelet<double>(2),
            new FejérKorovkinWavelet<double>(4),
            new ReverseBiorthogonalWavelet<double>(),
        };

        foreach (var wavelet in wavelets)
        {
            var coeffs = wavelet.GetWaveletCoefficients();
            Assert.True(coeffs.Length > 0,
                $"{wavelet.GetType().Name}.GetWaveletCoefficients() returned empty");
        }
    }

    #endregion
}
