using AiDotNet.Clustering.Evaluation;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Clustering;

/// <summary>
/// Cluster labels are arbitrary ids, so the cost of a metric must follow the number of samples and the
/// number of clusters present, never the largest id. The engine rewrite sized lookup tables and one-hot
/// masks by the label range, which turned two clusters labelled 0 and 500 000 into gigabytes of
/// intermediates (and an out-of-memory failure for larger ids).
/// </summary>
public class ClusterMetricLabelRangeRegressionTests
{
    private const int WideLabel = 500_000;

    /// <summary>Two well separated blobs of 20 points each; even rows are cluster 0.</summary>
    private static (Matrix<double> Data, Vector<double> Dense, Vector<double> Wide) TwoBlobs()
    {
        const int n = 40;
        var rng = new Random(11);
        var data = new Matrix<double>(n, 2);
        var dense = new Vector<double>(n);
        var wide = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            bool second = i % 2 == 1;
            data[i, 0] = (second ? 10.0 : 0.0) + rng.NextDouble();
            data[i, 1] = (second ? -5.0 : 0.0) + rng.NextDouble();
            dense[i] = second ? 1 : 0;
            wide[i] = second ? WideLabel : 0;
        }

        return (data, dense, wide);
    }

    /// <summary>Bytes allocated by the calling thread while running <paramref name="action"/>.</summary>
    private static long AllocatedBy(Action action)
    {
#if NET8_0_OR_GREATER
        long before = GC.GetAllocatedBytesForCurrentThread();
        action();
        return GC.GetAllocatedBytesForCurrentThread() - before;
#else
        // The per-thread counter is not available on .NET Framework 4.7.1; the values are still compared.
        action();
        return 0;
#endif
    }

    [Fact]
    public void InternalMetrics_WideLabelIds_MatchDenseLabelsWithoutRangeSizedIntermediates()
    {
        var (data, dense, wide) = TwoBlobs();
        var metrics = new IClusterMetric<double>[]
        {
            new SilhouetteScore<double>(),
            new CalinskiHarabaszIndex<double>(),
            new DaviesBouldinIndex<double>(),
            new WCSS<double>(),
            new BCSS<double>(),
        };

        foreach (var metric in metrics)
        {
            double expected = metric.Compute(data, dense);
            double actual = double.NaN;
            long allocated = AllocatedBy(() => actual = metric.Compute(data, wide));

            Assert.Equal(expected, actual, 12);
            Assert.True(allocated < 8L * 1024 * 1024,
                $"{metric.Name} allocated {allocated / (1024 * 1024)} MB for 40 samples in 2 clusters labelled 0 and {WideLabel}.");
        }
    }

    [Fact]
    public void ExternalMetrics_WideLabelIds_MatchDenseLabelsWithoutRangeSizedIntermediates()
    {
        // 20 000 000 is exactly representable in double and in single precision.
        const double big = 20_000_000;
        var truthDense = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0, 0.0, 1.0 });
        var truthWide = new Vector<double>(new[] { 0.0, 0.0, big, big, 0.0, big });
        var predicted = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0, 1.0, 1.0 });

        double expected = new AdjustedRandIndex<double>().Compute(truthDense, predicted);
        double actual = double.NaN;
        long allocated = AllocatedBy(() => actual = new AdjustedRandIndex<double>().Compute(truthWide, predicted));

        Assert.Equal(expected, actual, 12);
        Assert.True(allocated < 8L * 1024 * 1024,
            $"Adjusted Rand Index allocated {allocated / (1024 * 1024)} MB for 6 samples labelled 0 and {big}.");
    }

    [Fact]
    public void WcssPerCluster_WideLabelIds_KeyedByOriginalLabels()
    {
        var (data, dense, wide) = TwoBlobs();
        var wcss = new WCSS<double>();

        var expected = wcss.ComputePerCluster(data, dense);
        var actual = wcss.ComputePerCluster(data, wide);

        Assert.Equal(new[] { 0, WideLabel }, actual.Keys.OrderBy(key => key).ToArray());
        Assert.Equal(expected[0], actual[0], 12);
        Assert.Equal(expected[1], actual[WideLabel], 12);
    }

    [Fact]
    public void Silhouette_NoiseAndNonContiguousLabels_MatchesBruteForce()
    {
        // Labels 3 and 7 with two noise points: slots must be compacted without disturbing noise handling.
        var (data, _, _) = TwoBlobs();
        var labels = new Vector<double>(data.Rows);
        for (int i = 0; i < data.Rows; i++)
        {
            labels[i] = i < 2 ? -1 : i % 2 == 1 ? 7 : 3;
        }

        double actual = new SilhouetteScore<double>().Compute(data, labels);

        Assert.Equal(BruteForceSilhouette(data, labels), actual, 10);
    }

    [Fact]
    public void Silhouette_MoreRowsThanOneBlock_MatchesBruteForce()
    {
        // 2 100 rows exceeds one 2^22-element block of the row-blocked evaluation (2 048 x 2 048), so the
        // result crosses a block seam.
        const int n = 2100;
        var rng = new Random(5);
        var data = new Matrix<double>(n, 3);
        var labels = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            int cluster = i % 3;
            labels[i] = cluster;
            for (int j = 0; j < 3; j++)
            {
                data[i, j] = 4.0 * ((cluster + j) % 3) + rng.NextDouble();
            }
        }

        var metric = new SilhouetteScore<double>();
        double actual = metric.Compute(data, labels);
        var perSample = metric.ComputeSampleScores(data, labels);

        Assert.Equal(BruteForceSilhouette(data, labels), actual, 9);
        Assert.Equal(n, perSample.Length);
        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            sum += perSample[i];
        }

        Assert.Equal(actual, sum / n, 9);
    }

    private static double BruteForceSilhouette(Matrix<double> data, Vector<double> labels)
    {
        int n = data.Rows;
        var ids = new int[n];
        for (int i = 0; i < n; i++)
        {
            ids[i] = (int)labels[i];
        }

        var clusters = ids.Where(id => id >= 0).Distinct().ToArray();
        double total = 0;
        int valid = 0;
        for (int i = 0; i < n; i++)
        {
            if (ids[i] < 0)
            {
                continue;
            }

            valid++;
            var sums = new Dictionary<int, double>();
            var counts = new Dictionary<int, int>();
            foreach (int cluster in clusters)
            {
                sums[cluster] = 0;
                counts[cluster] = 0;
            }

            for (int j = 0; j < n; j++)
            {
                if (j == i || ids[j] < 0)
                {
                    continue;
                }

                double squared = 0;
                for (int c = 0; c < data.Columns; c++)
                {
                    double diff = data[i, c] - data[j, c];
                    squared += diff * diff;
                }

                sums[ids[j]] += Math.Sqrt(squared);
                counts[ids[j]]++;
            }

            if (counts[ids[i]] == 0)
            {
                continue;
            }

            double a = sums[ids[i]] / counts[ids[i]];
            double b = double.MaxValue;
            foreach (int cluster in clusters)
            {
                if (cluster != ids[i] && counts[cluster] > 0)
                {
                    b = Math.Min(b, sums[cluster] / counts[cluster]);
                }
            }

            if (b == double.MaxValue)
            {
                b = 0;
            }

            double max = Math.Max(a, b);
            if (max > 0)
            {
                total += (b - a) / max;
            }
        }

        return valid > 0 ? total / valid : 0;
    }
}
