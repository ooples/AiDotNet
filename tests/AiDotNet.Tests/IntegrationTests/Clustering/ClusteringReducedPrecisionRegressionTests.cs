using System;
using System.Reflection;
using AiDotNet.Clustering.AutoK;
using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Clustering.Probabilistic;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Clustering;

/// <summary>
/// Regression tests for the tensorized clustering helpers in reduced precision: counts, ranks and squares that
/// leave the range of half precision, epsilons that round to zero there, NaN from masked multiplies and argmin
/// over NaN. Each case failed on the element-wise sweep before the review fixes.
/// </summary>
public class ClusteringReducedPrecisionRegressionTests
{
    private static readonly INumericOperations<Half> HalfOps = MathHelper.GetNumericOperations<Half>();

    private static Half H(double value) => HalfOps.FromDouble(value);

    private static double D(Half value) => HalfOps.ToDouble(value);

    private static Matrix<Half> HalfMatrix(double[,] values)
    {
        var m = new Matrix<Half>(values.GetLength(0), values.GetLength(1));
        for (int i = 0; i < m.Rows; i++)
        {
            for (int j = 0; j < m.Columns; j++)
            {
                m[i, j] = H(values[i, j]);
            }
        }

        return m;
    }

    private static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);

    [Fact]
    public void NumericTolerance_ResolvesInHalfAndIsUnchangedInFloatAndDouble()
    {
        // 1e-10 is 0 in half precision; the tolerance is raised to the smallest normal half (2^-14).
        Assert.Equal(Math.Pow(2, -14), D(NumericTolerance.Of<Half>(1e-10)));
        Assert.Equal(1e-10, NumericTolerance.Of<double>(1e-10));
        Assert.Equal(1e-10f, NumericTolerance.Of<float>(1e-10));
    }

    [Fact]
    public void ClusterCounts_Half_CountPastTwoThousandFortyEight()
    {
        // A half-precision running count stops at 2048 (2048 + 1 rounds back to 2048).
        const int n = 3000;
        var labels = new Vector<Half>(n);
        var x = new Matrix<Half>(n, 1);

        var counts = ClusteringTensorOps.ClusterCounts(labels, 1);
        ClusteringTensorOps.ClusterSums(x, labels, 1, out var sumCounts);

        Assert.Equal(n, D(counts[0]));
        Assert.Equal(n, D(sumCounts[0]));
    }

    [Fact]
    public void DenseRanks_Half_RanksPastTwoThousandFortyEight()
    {
        const int n = 3000;
        var values = new Vector<Half>(n);
        for (int i = 0; i < n; i++)
        {
            values[i] = H((i - 1500) * 4.0);
        }

        var ranks = ClusteringTensorOps.DenseRanks(values);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(i, ranks[i]);
        }
    }

    [Fact]
    public void FromIntArray_WideIds_DoesNotAllocateRangeSizedTable()
    {
        var ids = new[] { 0, 20_000_000 };
        Vector<double> result = new Vector<double>(0);
#if NET8_0_OR_GREATER
        long before = GC.GetAllocatedBytesForCurrentThread();
        result = ClusteringTensorOps.FromIntArray<double>(ids);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(allocated < 8_000_000, $"FromIntArray allocated {allocated} bytes for two ids.");
#else
        result = ClusteringTensorOps.FromIntArray<double>(ids);
#endif
        Assert.Equal(2, result.Length);
        Assert.Equal(0.0, result[0]);
        Assert.Equal(20_000_000.0, result[1]);
    }

    [Fact]
    public void Distances_Half_DoNotOverflowWhenTheSquareDoes()
    {
        // 300^2 = 90000 is above the half maximum 65504; the distance itself is representable.
        var distances = ClusteringTensorOps.Distances(HalfMatrix(new double[,] { { 0 } }), HalfMatrix(new double[,] { { 300 } }));

        Assert.Equal(300.0, D(distances[0, 0]));
    }

    [Fact]
    public void StandardizeColumns_Half_ConstantAndWideColumns()
    {
        var x = HalfMatrix(new double[,] { { 5, 0 }, { 5, 1000 } });

        var standardized = ClusteringTensorOps.StandardizeColumns(x, out _, out var stds);

        Assert.Equal(1.0, D(stds[0]));
        Assert.Equal(500.0, D(stds[1]));
        for (int i = 0; i < 2; i++)
        {
            Assert.Equal(0.0, D(standardized[i, 0]));
        }

        Assert.Equal(-1.0, D(standardized[0, 1]));
        Assert.Equal(1.0, D(standardized[1, 1]));
    }

    [Fact]
    public void AssignedDistances_Half_InfiniteDistanceToAnotherCenterDoesNotPoisonTheRow()
    {
        // The distance to the other center (120000) is infinite in half precision; 0 * inf would be NaN.
        var x = HalfMatrix(new double[,] { { -60000 } });
        var centers = HalfMatrix(new double[,] { { -60000 }, { 60000 } });
        var labels = new Vector<Half>(1);

        var assigned = ClusteringTensorOps.AssignedDistances(new EuclideanDistance<Half>(), x, labels, centers);

        Assert.Equal(0.0, D(assigned[0]));
    }

    [Fact]
    public void ArgMin_NaNIsNeverTheMinimum()
    {
        var rows = new Matrix<double>(new double[,] { { double.NaN, 1, 2 }, { 3, double.NaN, 0 } });
        var vector = new Vector<double>(new[] { double.NaN, 3.0, 1.0 });

        Assert.Equal(new[] { 1, 2 }, ClusteringTensorOps.ArgMinRows(rows));
        Assert.Equal(2, ClusteringTensorOps.ArgMin(vector));
    }

    [Fact]
    public void KMeans_Half_SeparatesClustersWhoseSquaredNormsOverflow()
    {
        // Points around (+-300, +-300): ||x||^2 is ~180000, above the half maximum.
        const int perCluster = 20;
        var rng = RandomHelper.CreateSeededRandom(5);
        var values = new double[2 * perCluster, 2];
        for (int i = 0; i < 2 * perCluster; i++)
        {
            double center = i < perCluster ? -300 : 300;
            values[i, 0] = center + rng.NextDouble();
            values[i, 1] = center + rng.NextDouble();
        }

        var model = new KMeans<Half>(new KMeansOptions<Half> { NumClusters = 2, Seed = 3 });
        model.Train(HalfMatrix(values), new Vector<Half>(2 * perCluster));
        var labels = model.Labels ?? throw new InvalidOperationException("KMeans produced no labels.");

        for (int i = 1; i < perCluster; i++)
        {
            Assert.Equal(D(labels[0]), D(labels[i]));
            Assert.Equal(D(labels[perCluster]), D(labels[perCluster + i]));
        }

        Assert.NotEqual(D(labels[0]), D(labels[perCluster]));
    }

    [Fact]
    public void Mahalanobis_RoundingBelowZero_GivesZeroNotNaN()
    {
        // (1, -1) S (1, -1)^T = 2 - 2 (1 + 1e-12) = -2e-12 in floating point; the exact value is ~0.
        var inverse = new Matrix<double>(new double[,] { { 1, 1 + 1e-12 }, { 1 + 1e-12, 1 } });
        var metric = new MahalanobisDistance<double>(inverse);
        var a = new Vector<double>(new[] { 1.0, 0.0 });
        var b = new Vector<double>(new[] { 0.0, 1.0 });

        double scalar = metric.Compute(a, b);
        double pairwise = metric.ComputePairwise(
            new Matrix<double>(new double[,] { { 1, 0 } }), new Matrix<double>(new double[,] { { 0, 1 } }))[0, 0];

        Assert.True(scalar >= 0 && scalar < 1e-5, $"Compute returned {scalar}.");
        Assert.True(pairwise >= 0 && pairwise < 1e-5, $"ComputePairwise returned {pairwise}.");
    }

    [Fact]
    public void GaussianMixtureCholesky_ZeroPivot_StaysFinite()
    {
        var gmm = new GaussianMixtureModel<double>();
        var factor = gmm.Cholesky(new Matrix<double>(new double[,] { { 1, 0 }, { 0, 0 } }), 0.0, 0.0);

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                Assert.True(IsFinite(factor[i, j]), $"L[{i},{j}] = {factor[i, j]}.");
            }
        }

        Assert.Equal(1.0, factor[0, 0]);
        Assert.Equal(0.0, factor[0, 1]);
        Assert.Equal(0.0, factor[1, 0]);
    }

    [Fact]
    public void KMeans_Half_SerializeRoundTrip_KeepsCenters()
    {
        var x = HalfMatrix(new double[,] { { 0, 0 }, { 0.5, 0 }, { 10, 10 }, { 10.5, 10 } });
        var model = new KMeans<Half>(new KMeansOptions<Half> { NumClusters = 2, Seed = 1 });
        model.Train(x, new Vector<Half>(4));
        var centers = model.ClusterCenters ?? throw new InvalidOperationException("KMeans produced no centers.");

        var restored = new KMeans<Half>(new KMeansOptions<Half> { NumClusters = 2, Seed = 1 });
        restored.Deserialize(model.Serialize());
        var restoredCenters = restored.ClusterCenters ?? throw new InvalidOperationException("Centers were not restored.");

        Assert.Equal(centers.Rows, restoredCenters.Rows);
        for (int i = 0; i < centers.Rows; i++)
        {
            for (int j = 0; j < centers.Columns; j++)
            {
                Assert.Equal(D(centers[i, j]), D(restoredCenters[i, j]));
            }
        }
    }

    [Fact]
    public void AndersonDarling_Float_WellSeparatedPointIsFiniteAndMatchesDouble()
    {
        // 1 - F(7) is exactly 0 in single precision; the upper tail must be evaluated as F(-7).
        var z = new[] { -1.0, 0.0, 1.0, 7.0 };
        float single = new GMeans<float>().ComputeAndersonDarling(
            new Tensor<float>(Array.ConvertAll(z, v => (float)v), new[] { z.Length }), z.Length);
        double reference = new GMeans<double>().ComputeAndersonDarling(new Tensor<double>(z, new[] { z.Length }), z.Length);

        Assert.True(IsFinite(single), $"Anderson-Darling statistic was {single}.");
        Assert.True(Math.Abs(single - reference) <= 1e-3 * Math.Abs(reference) + 1e-4,
            $"float {single} vs double {reference}.");
    }

    [Fact]
    public void AndersonDarling_Half_ThreeHundredPointsIsFinite()
    {
        // The weighted sum reaches ~4 n^2 = 360000, above the half maximum.
        const int n = 300;
        var z = new Half[n];
        for (int i = 0; i < n; i++)
        {
            z[i] = H(-2.5 + 5.0 * i / (n - 1));
        }

        double ad = D(new GMeans<Half>().ComputeAndersonDarling(new Tensor<Half>(z, new[] { n }), n));

        Assert.True(IsFinite(ad), $"Anderson-Darling statistic was {ad}.");
    }

    [Fact]
    public void FuzzyCMeans_Half_MembershipAtACenterIsFinite()
    {
        var x = HalfMatrix(new double[,] { { -3, 0 }, { -3.5, 0.5 }, { -2.5, -0.5 }, { 3, 0 }, { 3.5, 0.5 }, { 2.5, -0.5 } });
        var model = new FuzzyCMeans<Half>(new FuzzyCMeansOptions<Half> { NumClusters = 2, Seed = 2 });
        model.Train(x, new Vector<Half>(x.Rows));
        var centers = model.ClusterCenters ?? throw new InvalidOperationException("FuzzyCMeans produced no centers.");

        var membership = model.PredictMembership(centers);

        for (int i = 0; i < membership.Rows; i++)
        {
            double rowSum = 0;
            for (int j = 0; j < membership.Columns; j++)
            {
                double value = D(membership[i, j]);
                Assert.True(IsFinite(value), $"membership[{i},{j}] = {value}.");
                rowSum += value;
            }

            Assert.True(Math.Abs(rowSum - 1.0) < 1e-2, $"Row {i} memberships sum to {rowSum}.");
        }
    }

    [Fact]
    public void AffinityPropagation_ResponsibilityOfANearTie_IsExact()
    {
        // r(0,1) = s(0,1) - max_{k != 1} s(0,k) = -5e-11 - 0. Treating values within 1e-10 of the row maximum as
        // tied maxima changes the excluded maximum.
        var similarity = new Tensor<double>(new[] { 0.0, -5e-11, -1.0, -1.0, 0.0, -1.0, -1.0, -1.0, 0.0 }, new[] { 3, 3 });
        var zeros = new Tensor<double>(new[] { 3, 3 });

        var r = new AffinityPropagation<double>().UpdateResponsibilities(similarity, zeros, zeros, 0.0, 1.0).ToArray();

        Assert.Equal(5e-11, r[0]);
        Assert.Equal(-5e-11, r[1]);
        Assert.Equal(-1.0, r[2]);
    }

    [Fact]
    public void MiniBatchKMeans_Half_LargeCenterCountStaysFinite()
    {
        // A center with 100000 earlier points: the count is infinite in half precision.
        var model = new MiniBatchKMeans<Half>(new MiniBatchKMeansOptions<Half> { NumClusters = 1 });
        var countsField = typeof(MiniBatchKMeans<Half>).GetField("_centerCounts", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("_centerCounts field not found.");
        countsField.SetValue(model, new[] { 100_000 });
        var update = typeof(MiniBatchKMeans<Half>).GetMethod("UpdateCentersWithBatch", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("UpdateCentersWithBatch not found.");
        var centers = HalfMatrix(new double[,] { { 1 } });

        update.Invoke(model, new object?[] { HalfMatrix(new double[,] { { 3 } }), centers });

        // (100000 * 1 + 3) / 100001
        double center = D(centers[0, 0]);
        Assert.True(IsFinite(center) && Math.Abs(center - 1.0) < 1e-2, $"Updated center was {center}.");
        Assert.Equal(100_001, ((int[])(countsField.GetValue(model) ?? Array.Empty<int>()))[0]);
    }
}
