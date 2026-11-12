using System;
using System.Threading.Tasks;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;
using AiDotNet.Preprocessing.FeatureSelection.Neural;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Reference tests for the third review round of the 08d feature-selection helpers and the sparse autoencoder:
/// the KL sparsity derivative at saturated units in Half, and the neighbour, shortest-path and distance helpers
/// where Half overflow met a mask multiply or a "max + 1" sentinel.
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

    // Three points all 4096 apart: each row's two nearest neighbours are the two other rows. The diagonal was pushed
    // to max + 1, which is 4096 itself in Half, so it tied with the real neighbours and row 0 picked itself.
    [Fact(Timeout = 60000)]
    public async Task KNearestMask_Half_NeverPicksTheRowItselfPast2048()
    {
        await Task.CompletedTask;
        var far = (Half)4096;
        var distances = new Tensor<Half>(
            new[] { (Half)0, far, far, far, (Half)0, far, far, far, (Half)0 }, new[] { 3, 3 });

        var mask = FeatureSelectionTensorOps.KNearestMask(distances, 2);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(i == j ? 0.0 : 1.0, (double)mask[i, j]);
    }

    // Floyd-Warshall on the chain 0 - 1 - 2 with edges of 30000 and an isolated node 3: d(0, 2) = 60000 and node 3 is
    // unreachable. The sentinel sum(lengths) + 1 = 120001 overflowed Half to infinity, and 0 * infinity made every
    // edge NaN.
    [Fact(Timeout = 60000)]
    public async Task ShortestPaths_Half_KeepsEdgesFiniteWhenTheLengthsSumPastHalfRange()
    {
        await Task.CompletedTask;
        const int n = 4;
        var lengths = new Half[n * n];
        var edges = new Half[n * n];
        foreach (var (a, b) in new[] { (0, 1), (1, 0), (1, 2), (2, 1) })
        {
            lengths[a * n + b] = (Half)30000;
            edges[a * n + b] = (Half)1;
        }

        var paths = FeatureSelectionTensorOps.ShortestPaths(
            new Tensor<Half>(lengths, new[] { n, n }), new Tensor<Half>(edges, new[] { n, n }), out var reachable);

        Assert.Equal(30000.0, (double)paths[0, 1]);
        Assert.Equal(60000.0, (double)paths[0, 2]);
        Assert.Equal(30000.0, (double)paths[2, 1]);
        Assert.Equal(0.0, (double)paths[1, 1]);
        Assert.True(double.IsPositiveInfinity((double)paths[0, 3]));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                Assert.Equal(i == j || (i < 3 && j < 3) ? 1.0 : 0.0, (double)reachable[i, j]);
    }

    // The diagonal of a squared-distance matrix is exactly 0 even where |a|^2 = 90000 overflows Half (a mask
    // multiply of an overflowed entry is 0 * infinity = NaN).
    [Fact(Timeout = 60000)]
    public async Task PairwiseSquaredDistances_Half_HasAnExactZeroDiagonal()
    {
        await Task.CompletedTask;
        var x = new Tensor<Half>(new[] { (Half)300, (Half)100 }, new[] { 2, 1 });

        var squared = FeatureSelectionTensorOps.PairwiseSquaredDistances(x);

        Assert.Equal(0.0, (double)squared[0, 0]);
        Assert.Equal(0.0, (double)squared[1, 1]);
    }
}
