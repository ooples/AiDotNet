using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Clustering;

/// <summary>
/// Regression tests for the Euclidean assignment step of <see cref="KMeans{T}"/>, which ranks centers by
/// <c>||x||^2 + ||c||^2 - 2 x.c</c> in the precision of T.
/// </summary>
public class KMeansAssignmentRegressionTests
{
    [Fact]
    public void SinglePrecision_ClustersFarFromOrigin_AreSeparated()
    {
        // Three clusters of spread 0.1, two units apart, around 10 000 in every coordinate. In single
        // precision ||x||^2 is ~4e8 with a rounding step of 32, far above the squared separation (16), so
        // the norm expansion on the raw data cannot tell the clusters apart; double precision could.
        const int n = 300;
        const int d = 4;
        const int k = 3;
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(17);
        var x = new Matrix<float>(n, d);
        for (int i = 0; i < n; i++)
        {
            int cluster = i % k;
            for (int j = 0; j < d; j++)
            {
                x[i, j] = (float)(10_000.0 + 2.0 * cluster + 0.1 * rng.NextDouble());
            }
        }

        var kmeans = new KMeans<float>(new KMeansOptions<float> { NumClusters = k, Seed = 7 });
        var labels = kmeans.FitPredict(x);

        var labelOfCluster = new int[k];
        for (int cluster = 0; cluster < k; cluster++)
        {
            labelOfCluster[cluster] = (int)labels[cluster];
        }

        Assert.Equal(k, labelOfCluster.Distinct().Count());
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(labelOfCluster[i % k], (int)labels[i]);
        }
    }

    [Fact]
    public void Predict_MoreRowsTimesClustersThanOneBlock_MatchesBruteForceNearestCenter()
    {
        // 70 000 rows x 64 centers exceeds one 2^22-element block of the row-blocked assignment.
        const int n = 70_000;
        const int d = 2;
        const int k = 64;
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(23);
        var x = new Matrix<double>(n, d);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                x[i, j] = 50.0 + 20.0 * rng.NextDouble();
            }
        }

        var kmeans = new KMeans<double>(new KMeansOptions<double>
        {
            NumClusters = k,
            Seed = 3,
            InitMethod = KMeansInitMethod.Random,
            NumInitializations = 1,
            MaxIterations = 2,
        });
        kmeans.Train(x);
        var centers = kmeans.ClusterCenters;
        Assert.NotNull(centers);

        var labels = kmeans.Predict(x);

        int mismatches = 0;
        for (int i = 0; i < n; i++)
        {
            int best = -1;
            double bestDistance = double.MaxValue;
            double second = double.MaxValue;
            for (int c = 0; c < centers.Rows; c++)
            {
                double squared = 0;
                for (int j = 0; j < d; j++)
                {
                    double diff = x[i, j] - centers[c, j];
                    squared += diff * diff;
                }

                if (squared < bestDistance)
                {
                    second = bestDistance;
                    bestDistance = squared;
                    best = c;
                }
                else if (squared < second)
                {
                    second = squared;
                }
            }

            // Skip rows that are numerically equidistant from two centers.
            if (second - bestDistance > 1e-9 && (int)labels[i] != best)
            {
                mismatches++;
            }
        }

        Assert.Equal(0, mismatches);
    }
}
