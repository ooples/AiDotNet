using System;
using System.Diagnostics;
using AiDotNet.Clustering.Density;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Clustering;

[Collection("NonParallelIntegration")]
public class ClusteringPerformanceIntegrationTests
{
    private const int MaxKMeansSeconds = 2;
    private const int MaxDbscanSeconds = 2;
    private const long MaxRetainedBytes = 128L * 1024 * 1024;

    [Fact]
    public void KMeans_CompletesWithinBudget()
    {
        var dataset = ClusteringTestData.CreateThreeClusterBlobs(pointsPerCluster: 50, spacing: 8.0);
        var options = new KMeansOptions<double>
        {
            NumClusters = 3,
            MaxIterations = 50,
            NumInitializations = 3,
            Seed = 42
        };

        var kmeans = new KMeans<double>(options);
        var (elapsed, retainedBytes) = Measure(() => kmeans.Train(dataset.Data));

        Assert.True(elapsed < TimeSpan.FromSeconds(MaxKMeansSeconds), $"KMeans took {elapsed.TotalSeconds:F2}s.");
        Assert.True(retainedBytes < MaxRetainedBytes, $"KMeans retained {retainedBytes / (1024.0 * 1024.0):F1} MB.");
        Assert.Equal(3, kmeans.NumClusters);
    }

    [Fact]
    public void DBSCAN_CompletesWithinBudget()
    {
        var dataset = ClusteringTestData.CreateTwoClusterBlobs(pointsPerCluster: 60, spacing: 8.0);
        var options = new DBSCANOptions<double>
        {
            Epsilon = 1.6,
            MinPoints = 3,
            Algorithm = NeighborAlgorithm.BruteForce
        };

        var dbscan = new DBSCAN<double>(options);
        var (elapsed, retainedBytes) = Measure(() => dbscan.Train(dataset.Data));

        Assert.True(elapsed < TimeSpan.FromSeconds(MaxDbscanSeconds), $"DBSCAN took {elapsed.TotalSeconds:F2}s.");
        Assert.True(retainedBytes < MaxRetainedBytes, $"DBSCAN retained {retainedBytes / (1024.0 * 1024.0):F1} MB.");
        Assert.True(dbscan.NumClusters >= 1);
    }

    private static (TimeSpan Elapsed, long RetainedBytes) Measure(Action action)
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        long before = GC.GetTotalMemory(true);
        var stopwatch = Stopwatch.StartNew();
        action();
        stopwatch.Stop();

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        long after = GC.GetTotalMemory(true);
        long retained = Math.Max(0, after - before);

        return (stopwatch.Elapsed, retained);
    }
}
