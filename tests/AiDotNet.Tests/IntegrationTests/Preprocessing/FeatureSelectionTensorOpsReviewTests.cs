using System;
using System.Threading.Tasks;
using AiDotNet.Preprocessing.FeatureSelection;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Reference tests for the shared filter feature-selection tensor helpers: counts past the Half integer range,
/// products that overflow Half, master's bin edges and random draw order, and the Wilcoxon tie term in float.
/// </summary>
public class FeatureSelectionTensorOpsReviewTests
{
    // 3000 rows all in code 0: a 0/1 sum in Half stops at 2048 (2048 + 1 rounds back to 2048).
    [Fact(Timeout = 60000)]
    public async Task CodeCounts_Half_CountPast2048()
    {
        await Task.CompletedTask;
        var codes = new Tensor<int>(new int[3000], new[] { 3000, 1 });

        var counts = FeatureSelectionTensorOps.CodeCounts<Half>(codes, 2);

        Assert.Equal(3000.0, (double)counts[0, 0]);
        Assert.Equal(0.0, (double)counts[0, 1]);
    }

    // The contingency table of one feature (all code 0) against a class one-hot with 3000 rows in class 1.
    [Fact(Timeout = 60000)]
    public async Task JointCounts_Half_CountPast2048()
    {
        await Task.CompletedTask;
        const int n = 4000;
        var codes = new Tensor<int>(new int[n], new[] { n, 1 });
        var classIndex = new int[n];
        for (int i = 0; i < 3000; i++)
            classIndex[i] = 1;

        var joint = FeatureSelectionTensorOps.JointCounts(
            codes, 1, FilterStatisticTensorOps.ClassOneHot<Half>(classIndex, 2));

        Assert.Equal(1000.0, (double)joint[0, 0, 0]);
        Assert.Equal(3000.0, (double)joint[0, 0, 1]);
    }

    // H(Y) for 3000 / 1000 samples: -(0.75 ln 0.75 + 0.25 ln 0.25). With class 1 counted as 2048 it was off.
    [Fact(Timeout = 60000)]
    public async Task DiscreteInformation_Half_TargetEntropyCountsPast2048()
    {
        await Task.CompletedTask;
        const int n = 4000;
        var target = new Half[n];
        for (int i = 0; i < 3000; i++)
            target[i] = (Half)1;
        var codes = new Tensor<int>(new int[n], new[] { n, 1 });

        var info = DiscreteInformation<Half>.ForTarget(codes, 1, new Vector<Half>(target));

        double expected = -(0.75 * Math.Log(0.75) + 0.25 * Math.Log(0.25));
        Assert.Equal(expected, (double)info.TargetEntropy, 2);
    }

    // Pearson chi-square of [[200, 100], [100, 200]]: every expected count is 300 * 300 / 600 = 150, so
    // chi2 = 4 * 50^2 / 150 = 66.67. The product 300 * 300 = 90000 overflowed Half and made the statistic NaN.
    [Fact(Timeout = 60000)]
    public async Task ContingencyChiSquare_Half_DoesNotOverflowTheExpectedCount()
    {
        await Task.CompletedTask;
        var observed = new Tensor<Half>(
            new[] { (Half)200, (Half)100, (Half)100, (Half)200 }, new[] { 1, 2, 2 });

        var chi2 = FeatureSelectionTensorOps.ContingencyChiSquare(observed);

        Assert.Equal(200.0 / 3.0, (double)chi2[0], 0);
    }

    // SURF pair weight with every pair a hit among 300 rows: -1 / (n * hits) = -1 / 90000. The product n * hits
    // overflowed Half to infinity and the weight became 0.
    [Fact(Timeout = 60000)]
    public async Task NearNeighbourPairWeight_Half_DoesNotOverflow()
    {
        await Task.CompletedTask;
        const int n = 300;
        var ones = new Half[n * n];
        for (int i = 0; i < ones.Length; i++)
            ones[i] = (Half)1;
        var hits = new Tensor<Half>(ones, new[] { n, n });
        var misses = new Tensor<Half>(new[] { n, n });

        var weight = FeatureSelectionTensorOps.NearNeighbourPairWeight(hits, misses);

        Assert.Equal(-1.0 / 90000.0, (double)weight[0, 0], 6);
        Assert.True((double)weight[0, 0] < 0);
    }

    // Master binned with ((x - min) / range) * bins: 0.9 / 9 * 10 = 1 exactly, so 0.9 is in bin 1. Multiplying by
    // the reciprocal (0.9 * (1 / 9) * 10 = 0.99999...) put it in bin 0.
    [Fact(Timeout = 60000)]
    public async Task ScaledBins_ValueOnABinEdge_KeepsMastersBin()
    {
        await Task.CompletedTask;
        var x = new Tensor<double>(new[] { 0.0, 0.9, 9.0 }, new[] { 3, 1 });

        var bins = FeatureSelectionTensorOps.ScaledBins(x, 10);

        Assert.Equal(new[] { 0, 1, 9 }, bins.ToArray());
    }

    // The first minimum of a 5000-column row whose minimum sits at columns 4000 and 4001 is column 4000.
    [Fact(Timeout = 60000)]
    public async Task FirstArgMinRows_Half_PicksTheFirstMinimumPastColumn2048()
    {
        await Task.CompletedTask;
        const int n = 5000;
        var values = new Half[n];
        for (int j = 0; j < n; j++)
            values[j] = (Half)1;
        values[4000] = (Half)0;
        values[4001] = (Half)0;

        var (index, found) = FeatureSelectionTensorOps.FirstArgMinRows(new Tensor<Half>(values, new[] { 1, n }));

        Assert.Equal(4000, index[0]);
        Assert.Equal(1.0, (double)found[0]);
    }

    // Power-iteration starts are random.NextDouble() - 0.5, start by start and row by row, as master drew them.
    [Fact(Timeout = 60000)]
    public async Task CenteredUniformStarts_DrawInMastersOrder()
    {
        await Task.CompletedTask;
        var starts = FeatureSelectionTensorOps.CenteredUniformStarts<double>(RandomHelper.CreateSeededRandom(7), 3, 4);
        var reference = RandomHelper.CreateSeededRandom(7);

        Assert.Equal(3, starts.Count);
        foreach (var start in starts)
        {
            Assert.Equal(new[] { 4, 1 }, start.Shape.ToArray());
            for (int i = 0; i < 4; i++)
                Assert.Equal(reference.NextDouble() - 0.5, start[i, 0], 12);
        }
    }

    // scipy.stats.wilcoxon(zero_method="wilcox") drops the 5000 zero differences. The tie term of the 100 that are
    // left (ten values tied ten times: 10 * (10^3 - 10) = 9900) used to be the column tie sum minus
    // (5000^3 - 5000), two numbers near 1.25e11 whose float rounding error (~8192) swamped it.
    [Fact(Timeout = 120000)]
    public async Task WilcoxonSignedRank_Float_TieTermSurvivesManyZeros()
    {
        await Task.CompletedTask;
        const int zeros = 5000;
        const int n = zeros + 100;
        var xd = new double[n];
        for (int i = 0; i < 100; i++)
            xd[zeros + i] = (1 + i % 10) * (i % 3 == 0 ? -1.0 : 1.0);
        var xf = new float[n];
        for (int i = 0; i < n; i++)
            xf[i] = (float)xd[i];

        var (wd, pd, md) = FeatureSelectionTensorOps.WilcoxonSignedRank(
            new Tensor<double>(xd, new[] { n, 1 }), new Tensor<double>(new[] { n }), 0.0);
        var (wf, pf, mf) = FeatureSelectionTensorOps.WilcoxonSignedRank(
            new Tensor<float>(xf, new[] { n, 1 }), new Tensor<float>(new[] { n }), 0.0);

        Assert.Equal(100.0, md[0]);
        Assert.Equal(100.0, (double)mf[0]);
        Assert.Equal(wd[0], (double)wf[0], 3);
        Assert.Equal(pd[0], (double)pf[0], 5);
    }

    // r = 1 for a feature equal to the target, 0 and 4 alternating over 300 rows: sxx = syy = 1200, and the
    // product sxx * syy = 1.44e6 overflowed Half, which made sqrt(sxx syy) infinite and r 0.
    [Fact(Timeout = 60000)]
    public async Task PearsonCorrelation_Half_DoesNotOverflowTheNormaliser()
    {
        await Task.CompletedTask;
        const int n = 300;
        var data = new Half[n, 1];
        var target = new Half[n];
        for (int i = 0; i < n; i++)
        {
            data[i, 0] = (Half)(4 * (i % 2));
            target[i] = (Half)(4 * (i % 2));
        }

        var selector = new AiDotNet.Preprocessing.FeatureSelection.Filter.PearsonCorrelation<Half>(nFeaturesToSelect: 1);
        selector.Fit(new Matrix<Half>(data), new Vector<Half>(target));

        Assert.NotNull(selector.Correlations);
        Assert.Equal(1.0, (double)selector.Correlations[0], 2);
    }

    private static Tensor<Half> HalfMatrix(double[] values, int rows, int columns)
    {
        var half = new Half[values.Length];
        for (int j = 0; j < values.Length; j++)
            half[j] = (Half)values[j];
        return new Tensor<Half>(half, new[] { rows, columns });
    }

    // Feature graph edges |r|: a constant column has r = 0 with every feature (master offsets its standard deviation
    // by 1e-10, which is 0 in Half, so 0 / 0 was NaN and the NaN passed the threshold), and 2 x_1 has r = 1 with x_1.
    [Fact(Timeout = 60000)]
    public async Task GraphBasedFS_Adjacency_Half_ConstantColumnHasNoEdges()
    {
        await Task.CompletedTask;
        var x = HalfMatrix(new[] { 1.0, 1.0, 2.0, 1.0, 2.0, 4.0, 1.0, 3.0, 6.0 }, 3, 3);
        var adjacency = AiDotNet.Preprocessing.FeatureSelection.Graph.GraphBasedFS<Half>.BuildAdjacencyMatrix(x, 0.5);
        Assert.Equal(0.0, (double)adjacency[0, 1]);
        Assert.Equal(0.0, (double)adjacency[0, 2]);
        Assert.Equal(0.0, (double)adjacency[1, 0]);
        Assert.Equal(1.0, (double)adjacency[1, 2], 2);
        Assert.Equal(0.0, (double)adjacency[1, 1]);
    }

    // Normalized Laplacian of W = 1 [300, 300]: every degree is 300, so L = I - W / 300. The product of degrees
    // 90000 overflowed Half before its root was taken, which made L = I.
    [Fact(Timeout = 60000)]
    public async Task SpectralClusteringFS_Laplacian_Half_DoesNotOverflowTheDegreeProduct()
    {
        await Task.CompletedTask;
        const int p = 300;
        var ones = new double[p * p];
        for (int j = 0; j < ones.Length; j++)
            ones[j] = 1;
        var laplacian = AiDotNet.Preprocessing.FeatureSelection.Graph.SpectralClusteringFS<Half>.NormalizedLaplacian(
            HalfMatrix(ones, p, p));
        Assert.Equal(-1.0 / p, (double)laplacian[0, 1], 4);
        Assert.Equal(1 - 1.0 / p, (double)laplacian[0, 0], 3);
    }

    // D^-1/2 A D^-1/2 with an isolated sample: its row and column are 0, the rest is A / sqrt(d_i d_j). The isolated
    // degree guard 1e-10 is 0 in Half (1 / sqrt(0) = inf), and the outer product of the scales overflows, both NaN.
    [Fact(Timeout = 60000)]
    public async Task MCFS_Propagation_Half_IsolatedSampleStaysZero()
    {
        await Task.CompletedTask;
        var propagation = AiDotNet.Preprocessing.FeatureSelection.Filter.Spectral.MCFS<Half>.Propagation(
            HalfMatrix(new[] { 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0 }, 3, 3));
        Assert.Equal(0.0, (double)propagation[0, 0]);
        Assert.Equal(0.0, (double)propagation[0, 1]);
        Assert.Equal(0.0, (double)propagation[1, 0]);
        Assert.Equal(0.5, (double)propagation[1, 2], 3);
        Assert.Equal(0.5, (double)propagation[2, 2], 3);
    }
}
