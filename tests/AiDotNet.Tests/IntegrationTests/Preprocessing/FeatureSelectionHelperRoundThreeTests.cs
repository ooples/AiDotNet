using System;
using System.Threading.Tasks;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;
using AiDotNet.Preprocessing.FeatureSelection.Neural;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Reference tests for the third review round of the 08d feature-selection helpers and the sparse autoencoder:
/// the KL sparsity derivative at saturated units in Half, and the pairwise squared distances where Half overflow
/// met a mask multiply.
/// </summary>
public class FeatureSelectionHelperRoundThreeTests
{
    // d/d rho^ of KL(rho || rho^) = -rho / rho^ + (1 - rho) / (1 - rho^) at rho^ clamped to [eps, 1 - eps] (Ng's
    // sparse autoencoder, with the clamp of common implementations). Units with mean activation 0 and 1 gave
    // -0.05 / 0 = -infinity and 0.95 / 0 = +infinity in Half, where the former 1e-10 guard is 0.
    [Fact(Timeout = 60000)]
    public async Task SparsityDelta_Half_StaysFiniteForSaturatedUnits()
    {
        await Task.CompletedTask;
        const double rho = 0.05;
        var mean = new Tensor<Half>(new[] { (Half)0, (Half)1, (Half)0.5 }, new[] { 3 });

        var delta = SparseAutoencoderSelector<Half>.SparsityDelta(mean, rho, 1.0);

        double eps = (double)SparseAutoencoderSelector<Half>.ActivationEpsilon();
        Assert.InRange(eps, 1e-6, 1e-3);
        // the upper clamp is the largest Half below 1, 1 - 2^-11
        double upper = 1 - Math.Pow(2, -11);
        double[] expected =
        {
            -rho / eps + (1 - rho) / (1 - eps),
            -rho / upper + (1 - rho) / (1 - upper),
            -rho / 0.5 + (1 - rho) / 0.5,
        };
        for (int i = 0; i < 3; i++)
        {
            double actual = (double)delta[i];
            Assert.False(double.IsInfinity(actual) || double.IsNaN(actual));
            Assert.InRange(actual, expected[i] - Math.Abs(expected[i]) * 0.01, expected[i] + Math.Abs(expected[i]) * 0.01);
        }
    }

    // In double the clamp margin is master's 1e-10, so the derivative is unchanged away from saturation.
    [Fact(Timeout = 60000)]
    public async Task ActivationEpsilon_Double_IsMastersGuard()
    {
        await Task.CompletedTask;
        Assert.Equal(1e-10, SparseAutoencoderSelector<double>.ActivationEpsilon());
    }

    // Rows 300 and 0: the squared distance 300^2 = 90000 is past Half's 65504, so the off-diagonal entries are
    // +infinity. The engine forms ||a||^2 + ||b||^2 - 2 a.b, so the diagonal of row 0 is inf + inf - inf = NaN, and
    // the former mask multiply (1 - I) kept it NaN (0 * NaN); the diagonal must be exactly 0.
    [Fact(Timeout = 60000)]
    public async Task PairwiseSquaredDistances_Half_HasAnExactZeroDiagonalWhenDistancesOverflow()
    {
        await Task.CompletedTask;
        var x = new Tensor<Half>(new[] { (Half)300, (Half)0 }, new[] { 2, 1 });

        var squared = FeatureSelectionTensorOps.PairwiseSquaredDistances(x);

        Assert.Equal(0.0, (double)squared[0, 0]);
        Assert.Equal(0.0, (double)squared[1, 1]);
        Assert.True(double.IsPositiveInfinity((double)squared[0, 1]));
        Assert.True(double.IsPositiveInfinity((double)squared[1, 0]));
    }
}
