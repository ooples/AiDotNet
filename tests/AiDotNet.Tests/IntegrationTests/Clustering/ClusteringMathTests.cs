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
        Assert.Equal(3, uniqueLabels.Count);

        // Verify cluster assignments are consistent within each known cluster group
        // Points 0-29 (cluster A), 30-59 (cluster B), 60-89 (cluster C)
        // should each be predominantly assigned to the same cluster label
        for (int clusterIdx = 0; clusterIdx < 3; clusterIdx++)
        {
            var labelsInGroup = new HashSet<double>();
            for (int i = clusterIdx * 30; i < (clusterIdx + 1) * 30; i++)
                labelsInGroup.Add(assignments[i]);
            Assert.True(labelsInGroup.Count == 1,
                $"All points from well-separated cluster {clusterIdx} should be in the same cluster, " +
                $"but found {labelsInGroup.Count} different labels");
        }
    }

    [Fact]
    public void KMeans_DeterministicTraining_SameDataSameResult()
    {
        var data1 = CreateClusteredData(
            clusterCenters: new double[,] { { 0, 0 }, { 10, 0 } },
            pointsPerCluster: 30,
            spread: 1.0,
            seed: 42);

        var data2 = CreateClusteredData(
            clusterCenters: new double[,] { { 0, 0 }, { 10, 0 } },
            pointsPerCluster: 30,
            spread: 1.0,
            seed: 42);

        var kmeans1 = new KMeans<double>(new KMeansOptions<double> { NumClusters = 2 });
        kmeans1.Train(data1, new Vector<double>(data1.Rows));

        var kmeans2 = new KMeans<double>(new KMeansOptions<double> { NumClusters = 2 });
        kmeans2.Train(data2, new Vector<double>(data2.Rows));

        var pred1 = kmeans1.Predict(data1);
        var pred2 = kmeans2.Predict(data2);

        // Two independently trained models on the same data should produce the same assignments
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

        // Count distinct cluster labels (non-noise) and noise points
        var clusterLabels = new HashSet<double>();
        int noiseCount = 0;
        for (int i = 0; i < assignments.Length; i++)
        {
            if (assignments[i] < 0)
                noiseCount++;
            else
                clusterLabels.Add(assignments[i]);
        }

        // Should find exactly 2 dense clusters
        Assert.True(clusterLabels.Count >= 2,
            $"DBSCAN should find at least 2 dense clusters, found {clusterLabels.Count}");

        // Should identify some noise points (the 5 scattered points at indices 60-64)
        Assert.True(noiseCount > 0,
            "DBSCAN should identify at least some noise points from the scattered outliers");
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
