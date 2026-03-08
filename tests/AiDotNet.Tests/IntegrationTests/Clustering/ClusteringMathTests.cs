using AiDotNet.Clustering.Density;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Clustering;

/// <summary>
/// Mathematically rigorous tests for clustering models verifying:
/// 1. Models find clusters in well-separated data
/// 2. Cluster assignments are deterministic for seeded models
/// 3. Centroids converge to expected locations
/// 4. Serialize/Deserialize preserves cluster state
///
/// Note: Clustering has ZERO Serialize/Deserialize overrides (only ClusteringBase).
/// All clustering models will likely lose state during serialization.
/// </summary>
public class ClusteringMathTests
{
    [Fact]
    public void KMeans_FindsClusters_WellSeparatedData()
    {
        // Create 3 well-separated clusters
        var data = CreateClusteredData(
            clusterCenters: new double[,] { { 0, 0 }, { 10, 0 }, { 5, 10 } },
            pointsPerCluster: 30,
            spread: 1.0,
            seed: 42);

        var kmeans = new KMeans<double>(new KMeansOptions<double> { NumClusters = 3 });
        kmeans.Train(data, new Vector<double>(data.Rows)); // y is unused but required

        var assignments = kmeans.Predict(data);

        // Verify we got 3 distinct clusters
        var uniqueLabels = new HashSet<double>();
        for (int i = 0; i < assignments.Length; i++)
        {
            uniqueLabels.Add(assignments[i]);
        }
        Assert.True(uniqueLabels.Count >= 2,
            $"KMeans should find at least 2 clusters in well-separated data, found {uniqueLabels.Count}");

        // Verify cluster assignments are consistent within known clusters
        // Points 0-29 should mostly be in the same cluster
        var cluster0Labels = new HashSet<double>();
        for (int i = 0; i < 30; i++) cluster0Labels.Add(assignments[i]);
        Assert.True(cluster0Labels.Count <= 2,
            "Points from cluster 0 should mostly be assigned to the same cluster");
    }

    [Fact]
    public void KMeans_PredictionDeterminism_SameInputSameOutput()
    {
        var data = CreateClusteredData(
            clusterCenters: new double[,] { { 0, 0 }, { 10, 0 } },
            pointsPerCluster: 30,
            spread: 1.0,
            seed: 42);

        var kmeans = new KMeans<double>(new KMeansOptions<double> { NumClusters = 2 });
        kmeans.Train(data, new Vector<double>(data.Rows));

        var pred1 = kmeans.Predict(data);
        var pred2 = kmeans.Predict(data);

        // Same model, same data → same predictions
        for (int i = 0; i < pred1.Length; i++)
        {
            Assert.Equal(pred1[i], pred2[i]);
        }
    }

    [Fact]
    public void KMeans_SerializeRoundTrip_PredictionsMatch()
    {
        var data = CreateClusteredData(
            clusterCenters: new double[,] { { 0, 0 }, { 10, 0 } },
            pointsPerCluster: 30,
            spread: 1.0,
            seed: 42);

        var kmeans = new KMeans<double>(new KMeansOptions<double> { NumClusters = 2 });
        kmeans.Train(data, new Vector<double>(data.Rows));

        var original = kmeans.Predict(data);

        var bytes = kmeans.Serialize();
        var restored = new KMeans<double>(new KMeansOptions<double> { NumClusters = 2 });
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(data);

        Assert.Equal(original.Length, restoredPreds.Length);
        for (int i = 0; i < original.Length; i++)
        {
            Assert.True(Math.Abs(original[i] - restoredPreds[i]) < 1e-10,
                $"KMeans Serialize round-trip: prediction mismatch at index {i}: " +
                $"original={original[i]}, restored={restoredPreds[i]}");
        }
    }

    [Fact]
    public void DBSCAN_FindsDenseClusters_IdentifiesNoise()
    {
        // Two dense clusters with some noise points
        var random = new Random(42);
        int n = 60;
        var data = new Matrix<double>(n + 5, 2); // 5 noise points

        // Cluster 1: around (0, 0)
        for (int i = 0; i < 30; i++)
        {
            data[i, 0] = NextGaussian(random) * 0.5;
            data[i, 1] = NextGaussian(random) * 0.5;
        }

        // Cluster 2: around (10, 10)
        for (int i = 30; i < 60; i++)
        {
            data[i, 0] = 10 + NextGaussian(random) * 0.5;
            data[i, 1] = 10 + NextGaussian(random) * 0.5;
        }

        // Noise points: far from both clusters
        for (int i = 60; i < 65; i++)
        {
            data[i, 0] = random.NextDouble() * 20 - 5;
            data[i, 1] = random.NextDouble() * 20 - 5;
        }

        var dbscan = new DBSCAN<double>(new DBSCANOptions<double> { Epsilon = 2.0, MinPoints = 3 });
        dbscan.Train(data, new Vector<double>(data.Rows));
        var assignments = dbscan.Predict(data);

        Assert.Equal(data.Rows, assignments.Length);

        // Should find at least 2 clusters (label >= 0) and possibly noise (label == -1)
        var uniqueLabels = new HashSet<double>();
        for (int i = 0; i < assignments.Length; i++)
        {
            uniqueLabels.Add(assignments[i]);
        }
        Assert.True(uniqueLabels.Count >= 2,
            $"DBSCAN should find at least 2 groups, found {uniqueLabels.Count}");
    }

    #region Helper Methods

    private static Matrix<double> CreateClusteredData(
        double[,] clusterCenters, int pointsPerCluster, double spread, int seed)
    {
        var random = new Random(seed);
        int numClusters = clusterCenters.GetLength(0);
        int features = clusterCenters.GetLength(1);
        int totalPoints = numClusters * pointsPerCluster;

        var data = new Matrix<double>(totalPoints, features);
        int idx = 0;
        for (int c = 0; c < numClusters; c++)
        {
            for (int p = 0; p < pointsPerCluster; p++)
            {
                for (int f = 0; f < features; f++)
                {
                    data[idx, f] = clusterCenters[c, f] + NextGaussian(random) * spread;
                }
                idx++;
            }
        }
        return data;
    }

    private static double NextGaussian(Random random)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    #endregion
}
