using System;
using System.Threading.Tasks;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Reference tests for the shared feature-selection tensor helpers: Half overflow in sums of squares and in
/// shortest-path sentinels, counts past Half's 2048, T-aware tolerances and LINQ-compatible tie order.
/// </summary>
public class FeatureSelectionTensorOpsTests
{
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

    // Top-2 of the masked entries {1, 2}. The old fill (min - 1) for the unmasked entry 0 equals 3e8 in float, so
    // entry 0 tied with entry 1 and won the tie by its lower index.
    [Fact(Timeout = 60000)]
    public async Task TopKWhere_RanksOnlyTheMaskedEntries()
    {
        await Task.CompletedTask;
        var scores = new Tensor<float>(new[] { 3e8f, 3e8f, 5e8f }, new[] { 3 });
        var mask = new Tensor<float>(new[] { 0f, 1f, 1f }, new[] { 3 });

        var selected = FeatureSelectionTensorOps.TopKWhere(scores, mask, 2);

        Assert.Equal(new[] { 1, 2 }, selected);
    }

    // Points 0, 1, 1, 2 on a line. Each row takes its 2 nearest other rows, ties at the 2nd distance going to the
    // lower index: row 0 -> {1, 2}, row 1 -> {2, 0}, row 2 -> {1, 0}, row 3 -> {1, 2}.
    [Fact(Timeout = 60000)]
    public async Task KNearestMask_BreaksTiesTowardTheLowerIndex()
    {
        await Task.CompletedTask;
        var points = new Tensor<double>(new[] { 0.0, 1.0, 1.0, 2.0 }, new[] { 4, 1 });
        var distances = FeatureSelectionTensorOps.PairwiseDistances(points);

        var mask = FeatureSelectionTensorOps.KNearestMask(distances, 2);

        double[,] expected =
        {
            { 0, 1, 1, 0 },
            { 1, 0, 1, 0 },
            { 1, 1, 0, 0 },
            { 0, 1, 1, 0 },
        };
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
                Assert.Equal(expected[i, j], mask[i, j]);
        }
    }

    // 2100 coincident points: every row has 2099 tied candidates and takes the 2051 lowest other indices. The
    // former tie rank was a TensorCumSum in T, which stops at 2048 in Half (2048 + 1 rounds to 2048), and k itself
    // rounds to 2052 in Half, so every row took more than 2051 neighbours (all 2099 with a Half running sum).
    [Fact(Timeout = 120000)]
    public async Task KNearestMask_Half_TakesExactlyKNeighboursAmongMoreThan2048Ties()
    {
        await Task.CompletedTask;
        const int n = 2100;
        const int k = 2051;
        var distances = new Tensor<Half>(new[] { n, n });

        var mask = FeatureSelectionTensorOps.KNearestMask(distances, k);

        for (int i = 0; i < n; i++)
        {
            int count = 0;
            for (int j = 0; j < n; j++)
            {
                int rank = j < i ? j : j - 1;
                double expected = j != i && rank < k ? 1.0 : 0.0;
                double actual = (double)mask[i, j];
                Assert.Equal(expected, actual);
                count += actual > 0 ? 1 : 0;
            }

            Assert.Equal(k, count);
        }
    }

    // A path 0 - 1 - 2 with edges of length 30000 and an isolated node 3. The former unreachable sentinel was the
    // sum of all edge lengths plus 1, 120001, which is +infinity in Half; the mask multiply then put 0 * infinity =
    // NaN on every edge. The paths are 30000 and 60000, and node 3 is unreachable (length 0, reachable 0).
    [Fact(Timeout = 60000)]
    public async Task ShortestPaths_Half_KeepsEdgeLengthsWhenTheirSumOverflows()
    {
        await Task.CompletedTask;
        const int n = 4;
        var lengthValues = new Half[n * n];
        var edgeValues = new Half[n * n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                lengthValues[i * n + j] = (Half)(i == j ? 0 : 30000);
                bool edge = Math.Abs(i - j) == 1 && i < 3 && j < 3;
                edgeValues[i * n + j] = (Half)(edge ? 1 : 0);
            }
        }

        var geodesic = FeatureSelectionTensorOps.ShortestPaths(
            new Tensor<Half>(lengthValues, new[] { n, n }), new Tensor<Half>(edgeValues, new[] { n, n }), out var reachable);

        double[,] expectedLength =
        {
            { 0, 30000, 60000, 0 },
            { 30000, 0, 30000, 0 },
            { 60000, 30000, 0, 0 },
            { 0, 0, 0, 0 },
        };
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Assert.Equal(expectedLength[i, j], (double)geodesic[i, j]);
                double expectedReachable = (i < 3 && j < 3) || i == j ? 1.0 : 0.0;
                Assert.Equal(expectedReachable, (double)reachable[i, j]);
            }
        }
    }

    // Two points 2e19 apart, each the other's only neighbour: the barycenter weight is 1. The squared difference
    // 4e38 is past float's 3.4e38, so the former Gram matrix and system were +infinity, the solve gave 0 and the
    // weights stayed 0; the differences are now divided by their largest magnitude first.
    [Fact(Timeout = 60000)]
    public async Task BarycenterWeights_Float_NormalizesWhenTheGramMatrixOverflows()
    {
        await Task.CompletedTask;
        var x = new Tensor<float>(new[] { 0f, 2e19f }, new[] { 2, 1 });

        var weights = FeatureSelectionTensorOps.BarycenterWeights(x, new[] { 1, 0 }, 1, 1e-3);

        Assert.Equal(0.0, weights[0, 0], 6);
        Assert.Equal(1.0, weights[0, 1], 6);
        Assert.Equal(1.0, weights[1, 0], 6);
        Assert.Equal(0.0, weights[1, 1], 6);
    }

    // In double the pre-scaling changes nothing: k = 2 neighbours at -1 and +2 of x_i = 0 (G = [[1, -2], [-2, 4]],
    // R = 1e-3 * 5) give w proportional to (G + R I)^-1 1, normalized to sum 1.
    [Fact(Timeout = 60000)]
    public async Task BarycenterWeights_Double_MatchesTheDirectSolve()
    {
        await Task.CompletedTask;
        var x = new Tensor<double>(new[] { 0.0, -1.0, 2.0 }, new[] { 3, 1 });

        var weights = FeatureSelectionTensorOps.BarycenterWeights(x, new[] { 1, 2, 0, 2, 0, 1 }, 2, 1e-3);

        const double r = 5e-3;
        double a = 1 + r, b = -2, d = 4 + r;
        double det = a * d - b * b;
        double w1 = (d - b) / det;
        double w2 = (a - b) / det;
        Assert.Equal(w1 / (w1 + w2), weights[0, 1], 10);
        Assert.Equal(w2 / (w1 + w2), weights[0, 2], 10);
        Assert.Equal(0.0, weights[0, 0], 10);
    }

    // Columns 0, 300, 0, 300 against 0, 600, 0, 600 (r = 1) and against its negation (r = -1). The former sums of
    // squares, 4 * 150^2 = 90000 and 360000, overflowed Half, so r was inf / inf = NaN.
    [Fact(Timeout = 60000)]
    public async Task ColumnCorrelations_Half_IsExactPastTheSquareOverflow()
    {
        await Task.CompletedTask;
        var a = new Tensor<Half>(new[] { (Half)0, (Half)300, (Half)0, (Half)300 }, new[] { 4, 1 });
        var b = new Tensor<Half>(
            new[] { (Half)0, (Half)0, (Half)600, (Half)(-600), (Half)0, (Half)0, (Half)600, (Half)(-600) },
            new[] { 4, 2 });

        var r = FeatureSelectionTensorOps.ColumnCorrelations(a, b);

        Assert.Equal(1.0, (double)r[0, 0], 2);
        Assert.Equal(-1.0, (double)r[0, 1], 2);
    }

    // Columns 0, 0.01 against themselves: sqrt(sxx) sqrt(syy) = 5e-5 is below the Half tolerance 2^-14 (the 1e-10
    // guard raised to Half's smallest normal), so the correlation is undefined and 0. The former guard was
    // FromDouble(1e-10), which is 0 in Half, so the subnormal denominator passed and r came out as 1.
    [Fact(Timeout = 60000)]
    public async Task ColumnCorrelations_Half_IsZeroBelowTheHalfTolerance()
    {
        await Task.CompletedTask;
        var a = new Tensor<Half>(new[] { (Half)0, (Half)0.01 }, new[] { 2, 1 });

        var r = FeatureSelectionTensorOps.ColumnCorrelations(a, a);

        Assert.Equal(0.0, (double)r[0, 0]);
    }

    // The tolerance only moves for Half: in double a denominator of 5e-5 is far above 1e-10 and r is 1.
    [Fact(Timeout = 60000)]
    public async Task ColumnCorrelations_Double_KeepsTheTenToTheMinusTenGuard()
    {
        await Task.CompletedTask;
        var a = new Tensor<double>(new[] { 0.0, 0.01 }, new[] { 2, 1 });

        var r = FeatureSelectionTensorOps.ColumnCorrelations(a, a);

        Assert.Equal(1.0, r[0, 0], 12);
    }

    // A centred column -300, 300 has population std 300. The former square of the raw values, 90000, overflowed
    // Half, so the std was +infinity.
    [Fact(Timeout = 60000)]
    public async Task CenteredColumnStd_Half_DoesNotOverflowPast256()
    {
        await Task.CompletedTask;
        var centered = new Tensor<Half>(new[] { (Half)(-300), (Half)300, (Half)0, (Half)0 }, new[] { 2, 2 });

        var std = FeatureSelectionTensorOps.CenteredColumnStd(centered);

        Assert.Equal(300.0, (double)std[0]);
        Assert.Equal(0.0, (double)std[1]);
    }

    // The column -300, 300 standardizes to -1, 1. With the former +infinity std (see above) it came out as 0, 0.
    [Fact(Timeout = 60000)]
    public async Task StandardizeColumnsOrKeep_Half_StandardizesColumnsPast256()
    {
        await Task.CompletedTask;
        var x = new Tensor<Half>(new[] { (Half)(-300), (Half)300 }, new[] { 2, 1 });

        var standardized = FeatureSelectionTensorOps.StandardizeColumnsOrKeep(x);

        Assert.Equal(-1.0, (double)standardized[0, 0]);
        Assert.Equal(1.0, (double)standardized[1, 0]);
    }

    // The column 0, 2e-5 has std 1e-5, below the Half tolerance 2^-14, so it is kept as it is. With the pre-scaled
    // std and the former FromDouble(1e-10) guard (0 in Half) it was standardized to -1, 1 (the code before the
    // pre-scaling kept it only because 1e-5 squared underflowed to 0).
    [Fact(Timeout = 60000)]
    public async Task StandardizeColumnsOrKeep_Half_KeepsAColumnWhoseStdIsBelowTheHalfTolerance()
    {
        await Task.CompletedTask;
        var x = new Tensor<Half>(new[] { (Half)0, (Half)2e-5 }, new[] { 2, 1 });

        var standardized = FeatureSelectionTensorOps.StandardizeColumnsOrKeep(x);

        Assert.Equal(0.0, (double)standardized[0, 0]);
        Assert.Equal((double)(Half)2e-5, (double)standardized[1, 0]);
    }
}
