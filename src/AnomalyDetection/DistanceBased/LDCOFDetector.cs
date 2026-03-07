using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.DistanceBased;

/// <summary>
/// Detects anomalies using LDCOF (Local Density Cluster-Based Outlier Factor).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> LDCOF combines clustering with density-based outlier detection.
/// It first clusters the data, then computes outlier scores based on how a point's density
/// compares to its cluster's density. Points in sparse regions of dense clusters are flagged.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Cluster the data using k-means
/// 2. Compute local density for each point
/// 3. Compare point density to cluster average density
/// 4. Large deviations indicate outliers
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When data has natural cluster structure
/// - For detecting local outliers within clusters
/// - When global methods miss cluster-specific anomalies
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Number of clusters: 8
/// - Number of neighbors (k): 10
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Amer, M., Goldstein, M. (2012). "Nearest-Neighbor and Clustering based
/// Anomaly Detection Algorithms for RapidMiner." Workshop on Open Source Data Mining Software.
/// </para>
/// </remarks>
public class LDCOFDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _numClusters;
    private readonly int _numNeighbors;
    private double[][]? _trainingData;
    private double[][]? _clusterCenters;
    private int[]? _clusterAssignments;
    private double[]? _clusterDensities;

    /// <summary>
    /// Gets the number of clusters.
    /// </summary>
    public int NumClusters => _numClusters;

    /// <summary>
    /// Gets the number of neighbors for density estimation.
    /// </summary>
    public int NumNeighbors => _numNeighbors;

    /// <summary>
    /// Creates a new LDCOF anomaly detector.
    /// </summary>
    /// <param name="numClusters">Number of clusters. Default is 8.</param>
    /// <param name="numNeighbors">Number of neighbors for density. Default is 10.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public LDCOFDetector(int numClusters = 8, int numNeighbors = 10,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (numClusters < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numClusters),
                "NumClusters must be at least 1. Recommended is 8.");
        }

        if (numNeighbors < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numNeighbors),
                "NumNeighbors must be at least 1. Recommended is 10.");
        }

        _numClusters = numClusters;
        _numNeighbors = numNeighbors;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        int d = X.Columns;

        // Convert to double array
        _trainingData = new double[n][];
        for (int i = 0; i < n; i++)
        {
            _trainingData[i] = new double[d];
            for (int j = 0; j < d; j++)
            {
                _trainingData[i][j] = NumOps.ToDouble(X[i, j]);
            }
        }

        // Run k-means clustering
        int effectiveClusters = Math.Min(_numClusters, n);
        (_clusterCenters, _clusterAssignments) = KMeansClustering(_trainingData, effectiveClusters);

        // Compute cluster densities
        _clusterDensities = ComputeClusterDensities();

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private (double[][] centers, int[] assignments) KMeansClustering(double[][] data, int k)
    {
        int n = data.Length;
        int d = data[0].Length;

        // Initialize centers randomly
        var centers = new double[k][];
        var indices = Enumerable.Range(0, n).OrderBy(_ => _random.NextDouble()).Take(k).ToArray();
        for (int i = 0; i < k; i++)
        {
            centers[i] = (double[])data[indices[i]].Clone();
        }

        var assignments = new int[n];
        int maxIterations = 100;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Assign points to nearest center
            bool changed = false;
            for (int i = 0; i < n; i++)
            {
                double minDist = double.MaxValue;
                int bestCluster = 0;

                for (int c = 0; c < k; c++)
                {
                    double dist = EuclideanDistance(data[i], centers[c]);
                    if (dist < minDist)
                    {
                        minDist = dist;
                        bestCluster = c;
                    }
                }

                if (assignments[i] != bestCluster)
                {
                    assignments[i] = bestCluster;
                    changed = true;
                }
            }

            if (!changed) break;

            // Update centers
            var counts = new int[k];
            var newCenters = new double[k][];
            for (int c = 0; c < k; c++)
            {
                newCenters[c] = new double[d];
            }

            for (int i = 0; i < n; i++)
            {
                int c = assignments[i];
                counts[c]++;
                for (int j = 0; j < d; j++)
                {
                    newCenters[c][j] += data[i][j];
                }
            }

            for (int c = 0; c < k; c++)
            {
                if (counts[c] > 0)
                {
                    for (int j = 0; j < d; j++)
                    {
                        newCenters[c][j] /= counts[c];
                    }
                    centers[c] = newCenters[c];
                }
            }
        }

        return (centers, assignments);
    }

    private double[] ComputeClusterDensities()
    {
        var clusterCenters = _clusterCenters;
        var trainingData = _trainingData;
        var clusterAssignments = _clusterAssignments;
        if (clusterCenters == null || trainingData == null || clusterAssignments == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        int k = clusterCenters.Length;
        var densities = new double[k];

        for (int c = 0; c < k; c++)
        {
            // Get points in this cluster
            var clusterPoints = trainingData
                .Select((p, i) => new { Point = p, Index = i })
                .Where(x => clusterAssignments[x.Index] == c)
                .Select(x => x.Point)
                .ToArray();

            if (clusterPoints.Length == 0)
            {
                densities[c] = 1.0;
                continue;
            }

            // Compute average k-distance within cluster
            double totalKDist = 0;
            int effectiveK = Math.Min(_numNeighbors, clusterPoints.Length - 1);

            foreach (var point in clusterPoints)
            {
                var distances = clusterPoints
                    .Where(p => p != point)
                    .Select(p => EuclideanDistance(point, p))
                    .OrderBy(d => d)
                    .ToArray();

                if (distances.Length > 0 && effectiveK > 0)
                {
                    totalKDist += distances[Math.Min(effectiveK - 1, distances.Length - 1)];
                }
            }

            // Average k-distance (inverse is density)
            double avgKDist = clusterPoints.Length > 0 ? totalKDist / clusterPoints.Length : 1.0;
            densities[c] = 1.0 / (avgKDist + 1e-10);
        }

        return densities;
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        var clusterCenters = _clusterCenters;
        var trainingData = _trainingData;
        var clusterDensities = _clusterDensities;
        if (clusterCenters == null || trainingData == null || clusterDensities == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var scores = new Vector<T>(X.Rows);
        int effectiveK = Math.Min(_numNeighbors, trainingData.Length - 1);

        for (int i = 0; i < X.Rows; i++)
        {
            var point = new double[X.Columns];
            for (int j = 0; j < X.Columns; j++)
            {
                point[j] = NumOps.ToDouble(X[i, j]);
            }

            // Find nearest cluster
            double minDist = double.MaxValue;
            int nearestCluster = 0;
            for (int c = 0; c < clusterCenters.Length; c++)
            {
                double dist = EuclideanDistance(point, clusterCenters[c]);
                if (dist < minDist)
                {
                    minDist = dist;
                    nearestCluster = c;
                }
            }

            // Compute local density (k-distance to training points)
            var distances = _trainingData!
                .Select(p => EuclideanDistance(point, p))
                .OrderBy(d => d)
                .ToArray();

            double kDist = effectiveK > 0 && effectiveK <= distances.Length
                ? distances[effectiveK - 1]
                : distances.LastOrDefault();
            double localDensity = 1.0 / (kDist + 1e-10);

            // LDCOF score: ratio of cluster density to local density
            double score = _clusterDensities![nearestCluster] / (localDensity + 1e-10);

            scores[i] = NumOps.FromDouble(score);
        }

        return scores;
    }

    private static double EuclideanDistance(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }
}
