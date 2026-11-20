using System;
using AiDotNet.Kernels;
using Xunit;

namespace AiDotNetTests.UnitTests.Kernels;

/// <summary>
/// Reference values (SciPy / closed forms) for kernel formulas corrected in the element-wise sweep.
/// </summary>
public class KernelReferenceValueTests
{
    private const double Tolerance = 1e-9;

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void Rff_HalfLargeInputs_MatchWidePhaseGeometry(int kind)
    {
        var kernel = new RFFKernel<Half>(2, 256, (RFFKernel<Half>.RFFKernelType)kind, seed: 17);
        var wide = new RFFKernel<double>(2, 256, (RFFKernel<double>.RFFKernelType)kind, seed: 17);
        var x = new Vector<Half>(new[] { (Half)60000, (Half)(-60000) });
        var expected = wide.GetFeatures(new Vector<double>(new[] { 60000.0, -60000.0 }));
        var actual = kernel.GetFeatures(x);
        for (int i = 0; i < actual.Length; i++) Assert.Equal((Half)expected[i], actual[i]);
        Assert.Equal((Half)wide.Calculate(new Vector<double>(new[] { 60000.0, -60000.0 }),
            new Vector<double>(new[] { 60000.0, -60000.0 })), kernel.Calculate(x, x));
    }

    [Fact]
    public void AngularKernels_Half_AvoidUnsupportedScalarConversionsAndNormOverflow()
    {
        var first = new Vector<Half>(new[] { (Half)1000 });
        Assert.Equal((Half)1, new NeuralNetworkKernel<Half>(order: 0).Calculate(first, first));
        Assert.Equal((Half)1, new DeepNeuralNetworkKernel<Half>(0, 1, 0, 2).Calculate(first, first));
        var zero = new Vector<Half>(new[] { (Half)0 });
        Assert.Equal((Half)0, new NeuralNetworkKernel<Half>().Calculate(zero, zero));
        var periodic = new ExpSineSquaredKernel<Half>();
        Assert.Equal((Half)1, periodic.Calculate(zero, zero));
    }

    [Fact]
    public void CholeskyLogDeterminant_Half_UsesSupportedWideSolver()
    {
        var matrix = new Matrix<Half>(new Half[,] { { (Half)4, (Half)0 }, { (Half)0, (Half)9 } });
        Assert.True(KernelVectorOps<Half>.TryCholeskyLogDeterminant(matrix, 1, out var actual));
        Assert.InRange((double)actual, Math.Log(36) - 0.005, Math.Log(36) + 0.005);
        var singular = new Matrix<Half>(new Half[,] { { (Half)1, (Half)1 }, { (Half)1, (Half)1 } });
        Assert.False(KernelVectorOps<Half>.TryCholeskyLogDeterminant(singular, 1, out _));
    }

    [Fact]
    public void KernelLinalg_Half_EigenAndTriangularSolveMatchClosedForms()
    {
        var matrix = AiDotNet.Tensors.LinearAlgebra.Tensor<Half>.FromMatrix(
            new Matrix<Half>(new Half[,] { { (Half)2, (Half)0 }, { (Half)0, (Half)3 } }));
        var (values, _) = AiDotNet.Helpers.LinalgDispatch<Half>.Eigh(matrix);
        Assert.Equal((Half)2, values.GetFlat(0));
        Assert.Equal((Half)3, values.GetFlat(1));
        var right = AiDotNet.Tensors.LinearAlgebra.Tensor<Half>.FromMatrix(
            new Matrix<Half>(new Half[,] { { (Half)4 }, { (Half)9 } }));
        var solved = AiDotNet.Helpers.LinalgDispatch<Half>.SolveTriangular(matrix, right, false);
        Assert.Equal((Half)2, solved.GetFlat(0));
        Assert.Equal((Half)3, solved.GetFlat(1));
    }

    [Fact]
    public void BesselKernel_OrderZero_MatchesScipyJ0()
    {
        // scipy.special.jv(0, 1.0) = 0.7651976865579666; kernel = J_0(r / sigma) / (r / sigma)^0
        var kernel = new BesselKernel<double>(order: 0.0, sigma: 1.0);
        var x1 = new Vector<double>(new[] { 0.0 });
        var x2 = new Vector<double>(new[] { 1.0 });

        Assert.Equal(0.7651976865579666, kernel.Calculate(x1, x2), 1e-10);
    }

    [Fact]
    public void BesselKernel_OrderOne_MatchesScipyJ1()
    {
        // scipy.special.jv(1, 2.0) / 2.0 = 0.2883624038784368
        var kernel = new BesselKernel<double>(order: 1.0, sigma: 1.0);
        var x1 = new Vector<double>(new[] { 0.0, 0.0 });
        var x2 = new Vector<double>(new[] { 2.0, 0.0 });

        Assert.Equal(0.2883624038784368, kernel.Calculate(x1, x2), 1e-10);
    }

    [Fact]
    public void GibbsKernel_TwoDimensions_UsesDimensionPowerOfNormalisation()
    {
        // Rasmussen & Williams eq. 4.32 with l(x) = 1 + x_0 in D = 2:
        // l1 = 1, l2 = 2, r^2 = 1 -> (2*1*2/5)^(2/2) * exp(-1/5) = 0.8 * exp(-0.2)
        var kernel = new GibbsKernel<double>(x => 1.0 + x[0]);
        var x1 = new Vector<double>(new[] { 0.0, 0.0 });
        var x2 = new Vector<double>(new[] { 1.0, 0.0 });

        Assert.Equal(0.8 * Math.Exp(-0.2), kernel.Calculate(x1, x2), Tolerance);
    }

    [Fact]
    public void SplineKernel_MatchesVapnikSplineFormula()
    {
        // Per dimension with m = min(x, y): 1 + xy + xy*m - (x + y)/2 * m^2 + m^3/3
        // (1, 2): 1 + 2 + 2 - 1.5 + 1/3; (0.5, 1): 1 + 0.5 + 0.25 - 0.1875 + 0.125/3
        var kernel = new SplineKernel<double>();
        var x1 = new Vector<double>(new[] { 1.0, 0.5 });
        var x2 = new Vector<double>(new[] { 2.0, 1.0 });

        double expected = (1 + 2 + 2 - 1.5 + 1.0 / 3.0) * (1 + 0.5 + 0.25 - 0.1875 + 0.125 / 3.0);
        Assert.Equal(expected, kernel.Calculate(x1, x2), Tolerance);
    }

    [Fact]
    public void SpectralDeltaKernel_SamePointInSeveralDimensions_ReturnsVariance()
    {
        // k(x, x) = sigma^2 for the spectral mixture product over dimensions (Wilson & Adams 2013).
        var kernel = new SpectralDeltaKernel<double>(frequency: 1.0, bandwidth: 0.1, variance: 2.5);
        var x = new Vector<double>(new[] { 0.3, -1.2, 4.0 });

        Assert.Equal(2.5, kernel.Calculate(x, x), Tolerance);
    }

    [Fact]
    public void BSpline_Cubic_IsCenteredAndSymmetric()
    {
        // Centered cubic B-spline: B3(0) = 2/3 and B3(+-0.5) = 23/48.
        var kernel = new BSplineKernel<double>(3, 1.0);
        var origin = new Vector<double>(new[] { 0.0, 0.0 });
        var shifted = new Vector<double>(new[] { 0.5, 0.0 });

        Assert.Equal(4.0 / 9.0, kernel.Calculate(origin, origin), 12);
        Assert.Equal(23.0 / 48.0 * 2.0 / 3.0, kernel.Calculate(origin, shifted), 12);
        Assert.Equal(kernel.Calculate(origin, shifted), kernel.Calculate(shifted, origin), 15);
    }
}
