using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.ClusterBased;

/// <summary>
/// Detects anomalies using Cluster-Based Local Outlier Factor (CBLOF).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> CBLOF combines clustering with local outlier detection.
/// It first clusters the data, then scores each point based on its distance to
/// its cluster center, weighted by the cluster size.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Cluster data into large and small clusters
/// 2. For large cluster points: score = cluster_size * distance_to_center
/// 3. For small cluster points: score = cluster_size * distance_to_nearest_large_cluster
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When anomalies are expected in small clusters or far from large clusters
/// - Faster than LOF for large datasets
/// - When cluster structure is meaningful
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - K (clusters): 8
/// - Alpha (large cluster threshold): 0.9 (90% of data)
/// - Beta (size threshold): 5 points
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: He, Z., et al. (2003). "Discovering Cluster-Based Local Outliers." Pattern Recognition Letters.
/// </para>
/// </remarks>
public class CBLOFDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _nClusters;
    private readonly double _alpha;
    private readonly int _beta;
    private Matrix<T>? _centroids;
    private int[]? _clusterSizes;
    private bool[]? _isLargeCluster;

    /// <summary>
    /// Gets the number of clusters.
    /// </summary>
    public int NClusters => _nClusters;

    /// <summary>
    /// Gets the alpha parameter (proportion for large clusters).
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Gets the beta parameter (minimum size for large clusters).
    /// </summary>
    public int Beta => _beta;

    /// <summary>
    /// Creates a new CBLOF anomaly detector.
    /// </summary>
    /// <param name="nClusters">Number of clusters. Default is 8.</param>
    /// <param name="alpha">
    /// Fraction of total points that large clusters should contain.
    /// Default is 0.9 (90%). Clusters are marked as large if they collectively
    /// contain at least alpha * total_points.
    /// </param>
    /// <param name="beta">
    /// Minimum number of points for a cluster to be considered large.
    /// Default is 5.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public CBLOFDetector(int nClusters = 8, double alpha = 0.9, int beta = 5,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (nClusters < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nClusters),
                "NClusters must be at least 1.");
        }

        if (alpha <= 0 || alpha > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(alpha),
                "Alpha must be between 0 (exclusive) and 1 (inclusive). Recommended is 0.9.");
        }

        if (beta < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(beta),
                "Beta must be at least 1. Recommended is 5.");
        }

        _nClusters = nClusters;
        _alpha = alpha;
        _beta = beta;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Rows < _nClusters)
        {
            throw new ArgumentException(
                $"Number of samples ({X.Rows}) must be at least nClusters ({_nClusters}).",
                nameof(X));
        }

        // Run K-Means clustering
        var (centroids, assignments) = RunKMeans(X);
        _centroids = centroids;

        // Compute cluster sizes
        _clusterSizes = new int[_nClusters];
        for (int i = 0; i < X.Rows; i++)
        {
            _clusterSizes[assignments[i]]++;
        }

        // Determine large and small clusters
        _isLargeCluster = DetermineLargeClusters(X.Rows);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
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

        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            var point = new Vector<T>(X.Columns);
            for (int j = 0; j < X.Columns; j++)
            {
                point[j] = X[i, j];
            }

            // Find nearest cluster and distance
            int nearestCluster = 0;
            double nearestDist = double.MaxValue;

            var centroids = _centroids;
            var isLargeCluster = _isLargeCluster;
            var clusterSizes = _clusterSizes;
            if (centroids == null || isLargeCluster == null || clusterSizes == null)
            {
                throw new InvalidOperationException("Model not properly fitted.");
            }

            for (int c = 0; c < _nClusters; c++)
            {
                var centroid = new Vector<T>(centroids.Columns);
                for (int j = 0; j < centroids.Columns; j++)
                {
                    centroid[j] = centroids[c, j];
                }

                double dist = NumOps.ToDouble(StatisticsHelper<T>.EuclideanDistance(point, centroid));
                if (dist < nearestDist)
                {
                    nearestDist = dist;
                    nearestCluster = c;
                }
            }

            double cblof;
            if (isLargeCluster[nearestCluster])
            {
                // Large cluster: CBLOF = size * distance_to_centroid
                cblof = clusterSizes[nearestCluster] * nearestDist;
            }
            else
            {
                // Small cluster: CBLOF = size * distance_to_nearest_large_cluster
                double minLargeDist = double.MaxValue;
                for (int c = 0; c < _nClusters; c++)
                {
                    if (!isLargeCluster[c]) continue;

                    var centroid = new Vector<T>(centroids.Columns);
                    for (int j = 0; j < centroids.Columns; j++)
                    {
                        centroid[j] = centroids[c, j];
                    }

                    double dist = NumOps.ToDouble(StatisticsHelper<T>.EuclideanDistance(point, centroid));
                    if (dist < minLargeDist)
                    {
                        minLargeDist = dist;
                    }
                }

                cblof = clusterSizes[nearestCluster] * minLargeDist;
            }

            scores[i] = NumOps.FromDouble(cblof);
        }

        return scores;
    }

    private bool[] DetermineLargeClusters(int totalPoints)
    {
        var isLarge = new bool[_nClusters];

        // Sort clusters by size (descending)
        var sortedClusters = Enumerable.Range(0, _nClusters)
            .OrderByDescending(c => _clusterSizes![c])
            .ToList();

        // Mark clusters as large until we reach alpha * total_points
        int cumulativeSize = 0;
        int threshold = (int)(_alpha * totalPoints);

        foreach (int c in sortedClusters)
        {
            if (_clusterSizes![c] >= _beta && cumulativeSize < threshold)
            {
                isLarge[c] = true;
                cumulativeSize += _clusterSizes[c];
            }
        }

        // Ensure at least one large cluster
        if (!isLarge.Any(l => l) && sortedClusters.Count > 0)
        {
            isLarge[sortedClusters[0]] = true;
        }

        return isLarge;
    }

    private (Matrix<T> centroids, int[] assignments) RunKMeans(Matrix<T> X)
    {
        var random = new Random(_randomSeed);
        int n = X.Rows;
        int d = X.Columns;

        // Initialize centroids randomly
        var centroids = new Matrix<T>(_nClusters, d);
        var selectedIndices = new HashSet<int>();

        for (int c = 0; c < _nClusters; c++)
        {
            int idx;
            do
            {
                idx = random.Next(n);
            } while (selectedIndices.Contains(idx));

            selectedIndices.Add(idx);

            for (int j = 0; j < d; j++)
            {
                centroids[c, j] = X[idx, j];
            }
        }

        var assignments = new int[n];
        int maxIterations = 100;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Assignment step
            bool changed = false;
            for (int i = 0; i < n; i++)
            {
                int nearest = 0;
                double minDist = double.MaxValue;

                var point = new Vector<T>(d);
                for (int j = 0; j < d; j++)
                {
                    point[j] = X[i, j];
                }

                for (int c = 0; c < _nClusters; c++)
                {
                    var centroid = new Vector<T>(d);
                    for (int j = 0; j < d; j++)
                    {
                        centroid[j] = centroids[c, j];
                    }

                    double dist = NumOps.ToDouble(StatisticsHelper<T>.EuclideanDistance(point, centroid));
                    if (dist < minDist)
                    {
                        minDist = dist;
                        nearest = c;
                    }
                }

                if (assignments[i] != nearest)
                {
                    assignments[i] = nearest;
                    changed = true;
                }
            }

            if (!changed) break;

            // Update step
            var newCentroids = new Matrix<T>(_nClusters, d);
            var counts = new int[_nClusters];

            for (int i = 0; i < n; i++)
            {
                int cluster = assignments[i];
                counts[cluster]++;
                for (int j = 0; j < d; j++)
                {
                    newCentroids[cluster, j] = NumOps.Add(newCentroids[cluster, j], X[i, j]);
                }
            }

            for (int c = 0; c < _nClusters; c++)
            {
                if (counts[c] > 0)
                {
                    for (int j = 0; j < d; j++)
                    {
                        newCentroids[c, j] = NumOps.Divide(newCentroids[c, j], NumOps.FromDouble(counts[c]));
                    }
                }
                else
                {
                    // Re-initialize empty cluster
                    int randomIdx = random.Next(n);
                    for (int j = 0; j < d; j++)
                    {
                        newCentroids[c, j] = X[randomIdx, j];
                    }
                }
            }

            centroids = newCentroids;
        }

        return (centroids, assignments);
    }
}
