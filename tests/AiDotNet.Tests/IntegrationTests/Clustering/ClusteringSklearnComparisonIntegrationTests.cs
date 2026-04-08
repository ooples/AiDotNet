using System;
using System.Collections.Generic;
using System.IO;
using AiDotNet.Clustering.Density;
using AiDotNet.Clustering.Evaluation;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Clustering;

public class ClusteringSklearnComparisonIntegrationTests
{
    private static readonly Lazy<SklearnReferenceData> Reference = new(SklearnReferenceData.Load);

    [Fact]
    public void KMeans_MatchesSklearnReference()
    {
        var reference = Reference.Value;
        var options = new KMeansOptions<double>
        {
            NumClusters = 2,
            InitMethod = KMeansInitMethod.KMeansPlusPlus,
            NumInitializations = 10,
            Seed = 42,
            MaxIterations = 300
        };

        var kmeans = new KMeans<double>(options);
        kmeans.Train(reference.TwoClusterData);

        var labels = ClusteringTestHelpers.RequireNotNull(kmeans.Labels, "Labels");
        var centers = ClusteringTestHelpers.RequireNotNull(kmeans.ClusterCenters, "ClusterCenters");
        if (!(kmeans.Inertia is double inertia))
        {
            throw new InvalidOperationException("Inertia was null.");
        }

        Assert.Equal(reference.KMeansLabels.Length, labels.Length);
        Assert.Equal(reference.KMeansCenters.Rows, centers.Rows);

        double ari = new AdjustedRandIndex<double>().Compute(reference.KMeansLabels, labels);
        Assert.True(ari > 0.999, $"ARI {ari:F6} is below expected match threshold.");

        var orderedCenters = SortCenters(centers);
        var orderedReferenceCenters = SortCenters(reference.KMeansCenters);

        const double centerTolerance = 1e-3;
        for (int i = 0; i < orderedCenters.Count; i++)
        {
            Assert.Equal(orderedReferenceCenters[i].X, orderedCenters[i].X, centerTolerance);
            Assert.Equal(orderedReferenceCenters[i].Y, orderedCenters[i].Y, centerTolerance);
        }

        Assert.Equal(reference.KMeansInertia, inertia, 1e-2);
    }

    [Fact]
    public void ClusterMetrics_MatchSklearnReference()
    {
        var reference = Reference.Value;
        var metrics = new ClusterMetrics<double>().Evaluate(reference.TwoClusterData, reference.KMeansLabels);

        Assert.Equal(reference.Silhouette, metrics.Silhouette, 1e-4);
        Assert.Equal(reference.DaviesBouldin, metrics.DaviesBouldin, 1e-4);
        Assert.Equal(reference.CalinskiHarabasz, metrics.CalinskiHarabasz, 1e-4);
    }

    [Fact]
    public void DBSCAN_MatchesSklearnReference()
    {
        var reference = Reference.Value;
        var options = new DBSCANOptions<double>
        {
            Epsilon = 1.6,
            MinPoints = 3,
            Algorithm = NeighborAlgorithm.BruteForce
        };

        var dbscan = new DBSCAN<double>(options);
        dbscan.Train(reference.WithOutlierData);

        var labels = ClusteringTestHelpers.RequireNotNull(dbscan.Labels, "Labels");
        Assert.Equal(reference.DbscanLabels.Length, labels.Length);

        double ari = new AdjustedRandIndex<double>().Compute(reference.DbscanLabels, labels);
        Assert.True(ari > 0.999, $"ARI {ari:F6} is below expected match threshold.");

        Assert.Equal(reference.DbscanNumClusters, dbscan.NumClusters);
        Assert.Equal(reference.DbscanNumNoise, dbscan.GetNoiseCount());
    }

    private static List<(double X, double Y)> SortCenters(Matrix<double> centers)
    {
        var list = new List<(double X, double Y)>(centers.Rows);
        for (int i = 0; i < centers.Rows; i++)
        {
            list.Add((centers[i, 0], centers[i, 1]));
        }

        list.Sort((a, b) =>
        {
            int cmp = a.X.CompareTo(b.X);
            return cmp != 0 ? cmp : a.Y.CompareTo(b.Y);
        });

        return list;
    }
}

internal sealed class SklearnReferenceData
{
    private const string ReferenceFileName = "sklearn_clustering_reference.json";

    private SklearnReferenceData(
        Matrix<double> twoClusterData,
        Matrix<double> withOutlierData,
        Vector<double> kmeansLabels,
        Matrix<double> kmeansCenters,
        double kmeansInertia,
        Vector<double> dbscanLabels,
        int dbscanNumClusters,
        int dbscanNumNoise,
        double silhouette,
        double daviesBouldin,
        double calinskiHarabasz)
    {
        TwoClusterData = twoClusterData;
        WithOutlierData = withOutlierData;
        KMeansLabels = kmeansLabels;
        KMeansCenters = kmeansCenters;
        KMeansInertia = kmeansInertia;
        DbscanLabels = dbscanLabels;
        DbscanNumClusters = dbscanNumClusters;
        DbscanNumNoise = dbscanNumNoise;
        Silhouette = silhouette;
        DaviesBouldin = daviesBouldin;
        CalinskiHarabasz = calinskiHarabasz;
    }

    public Matrix<double> TwoClusterData { get; }
    public Matrix<double> WithOutlierData { get; }
    public Vector<double> KMeansLabels { get; }
    public Matrix<double> KMeansCenters { get; }
    public double KMeansInertia { get; }
    public Vector<double> DbscanLabels { get; }
    public int DbscanNumClusters { get; }
    public int DbscanNumNoise { get; }
    public double Silhouette { get; }
    public double DaviesBouldin { get; }
    public double CalinskiHarabasz { get; }

    public static SklearnReferenceData Load()
    {
        string path = ResolveReferencePath();
        var root = JObject.Parse(File.ReadAllText(path));

        var datasets = RequireObject(root, "datasets");
        var twoClusterData = ReadMatrix(datasets["two_cluster_blobs_12_12"]?["data"], "datasets.two_cluster_blobs_12_12.data");
        var withOutlierData = ReadMatrix(datasets["with_outlier"]?["data"], "datasets.with_outlier.data");

        var kmeansRoot = RequireObject(root, "kmeans");
        var kmeansData = RequireObject(kmeansRoot, "two_cluster_blobs_12_12");
        var kmeansLabels = ToVector(ReadIntArray(kmeansData["labels"], "kmeans.two_cluster_blobs_12_12.labels"));
        var kmeansCenters = ReadMatrix(kmeansData["centers"], "kmeans.two_cluster_blobs_12_12.centers");
        double kmeansInertia = ReadDouble(kmeansData["inertia"], "kmeans.two_cluster_blobs_12_12.inertia");

        var dbscanRoot = RequireObject(root, "dbscan");
        var dbscanData = RequireObject(dbscanRoot, "with_outlier_eps_1_6_min_3");
        var dbscanLabels = ToVector(ReadIntArray(dbscanData["labels"], "dbscan.with_outlier_eps_1_6_min_3.labels"));
        int dbscanNumClusters = ReadInt(dbscanData["num_clusters"], "dbscan.with_outlier_eps_1_6_min_3.num_clusters");
        int dbscanNumNoise = ReadInt(dbscanData["num_noise"], "dbscan.with_outlier_eps_1_6_min_3.num_noise");

        var metricsRoot = RequireObject(root, "metrics");
        var metricsData = RequireObject(metricsRoot, "two_cluster_blobs_12_12");
        double silhouette = ReadDouble(metricsData["silhouette"], "metrics.two_cluster_blobs_12_12.silhouette");
        double daviesBouldin = ReadDouble(metricsData["davies_bouldin"], "metrics.two_cluster_blobs_12_12.davies_bouldin");
        double calinskiHarabasz = ReadDouble(metricsData["calinski_harabasz"], "metrics.two_cluster_blobs_12_12.calinski_harabasz");

        return new SklearnReferenceData(
            twoClusterData,
            withOutlierData,
            kmeansLabels,
            kmeansCenters,
            kmeansInertia,
            dbscanLabels,
            dbscanNumClusters,
            dbscanNumNoise,
            silhouette,
            daviesBouldin,
            calinskiHarabasz);
    }

    private static string ResolveReferencePath()
    {
        string baseDir = AppContext.BaseDirectory;
        string outputPath = Path.Combine(baseDir, "IntegrationTests", "Clustering", "ReferenceData", ReferenceFileName);
        if (File.Exists(outputPath))
        {
            return outputPath;
        }

        string? repoRoot = FindRepoRoot(baseDir);
        if (repoRoot is not null)
        {
            string sourcePath = Path.Combine(repoRoot, "tests", "AiDotNet.Tests", "IntegrationTests", "Clustering", "ReferenceData", ReferenceFileName);
            if (File.Exists(sourcePath))
            {
                return sourcePath;
            }
        }

        throw new FileNotFoundException($"Unable to locate {ReferenceFileName} from {baseDir}.");
    }

    private static string? FindRepoRoot(string startPath)
    {
        var current = new DirectoryInfo(startPath);
        while (current is not null)
        {
            if (File.Exists(Path.Combine(current.FullName, "AiDotNet.sln")))
            {
                return current.FullName;
            }

            current = current.Parent;
        }

        return null;
    }

    private static JObject RequireObject(JObject parent, string name)
    {
        var token = parent[name];
        if (token is JObject obj)
        {
            return obj;
        }

        throw new InvalidDataException($"Missing object '{name}'.");
    }

    private static Matrix<double> ReadMatrix(JToken? token, string name)
    {
        if (token is not JArray rows)
        {
            throw new InvalidDataException($"Expected array for '{name}'.");
        }

        if (rows.Count == 0)
        {
            throw new InvalidDataException($"Expected non-empty array for '{name}'.");
        }

        int columns = (rows[0] as JArray)?.Count ?? 0;
        if (columns == 0)
        {
            throw new InvalidDataException($"Expected non-empty rows for '{name}'.");
        }

        var matrix = new Matrix<double>(rows.Count, columns);
        for (int i = 0; i < rows.Count; i++)
        {
            if (rows[i] is not JArray row)
            {
                throw new InvalidDataException($"Row {i} in '{name}' is not an array.");
            }

            if (row.Count != columns)
            {
                throw new InvalidDataException($"Row {i} in '{name}' has {row.Count} columns, expected {columns}.");
            }

            for (int j = 0; j < columns; j++)
            {
                matrix[i, j] = ReadDouble(row[j], $"{name}[{i}][{j}]");
            }
        }

        return matrix;
    }

    private static int[] ReadIntArray(JToken? token, string name)
    {
        if (token is not JArray array)
        {
            throw new InvalidDataException($"Expected array for '{name}'.");
        }

        var values = new int[array.Count];
        for (int i = 0; i < array.Count; i++)
        {
            values[i] = ReadInt(array[i], $"{name}[{i}]");
        }

        return values;
    }

    private static int ReadInt(JToken? token, string name)
    {
        if (token is null || token.Type == JTokenType.Null)
        {
            throw new InvalidDataException($"Missing integer value for '{name}'.");
        }

        return token.Value<int>();
    }

    private static double ReadDouble(JToken? token, string name)
    {
        if (token is null || token.Type == JTokenType.Null)
        {
            throw new InvalidDataException($"Missing numeric value for '{name}'.");
        }

        return token.Value<double>();
    }

    private static Vector<double> ToVector(int[] values)
    {
        var vector = new Vector<double>(values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            vector[i] = values[i];
        }

        return vector;
    }
}
