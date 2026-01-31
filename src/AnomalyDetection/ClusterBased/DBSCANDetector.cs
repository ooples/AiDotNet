using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.ClusterBased;

/// <summary>
/// Detects anomalies using DBSCAN clustering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
/// groups together points that are closely packed and marks points in low-density regions
/// as noise (anomalies). Unlike k-means, DBSCAN can find arbitrarily shaped clusters.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Define a neighborhood radius (epsilon) and minimum points (minPts)
/// 2. A core point has at least minPts within epsilon distance
/// 3. Points reachable from core points form clusters
/// 4. Points not reachable from any core point are noise (anomalies)
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When clusters have irregular shapes
/// - When you don't know the number of clusters
/// - When anomalies are expected to be in low-density regions
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Epsilon: Estimated from k-distance graph (auto by default)
/// - MinPts: 2 * dimensions (typical heuristic)
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Ester, M., et al. (1996). "A Density-Based Algorithm for Discovering Clusters
/// in Large Spatial Databases with Noise." KDD.
/// </para>
/// </remarks>
public class DBSCANDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double? _epsilon;
    private readonly int? _minPts;
    private double _fittedEpsilon;
    private int _fittedMinPts;
    private int[]? _clusterLabels;
    private Matrix<T>? _trainingData;

    /// <summary>
    /// Gets the epsilon (neighborhood radius) parameter.
    /// </summary>
    public double Epsilon => _fittedEpsilon;

    /// <summary>
    /// Gets the minimum points parameter.
    /// </summary>
    public int MinPts => _fittedMinPts;

    /// <summary>
    /// Creates a new DBSCAN anomaly detector.
    /// </summary>
    /// <param name="epsilon">
    /// Neighborhood radius. If null, estimated automatically from the data.
    /// </param>
    /// <param name="minPts">
    /// Minimum number of points to form a dense region. If null, uses 2 * dimensions.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public DBSCANDetector(double? epsilon = null, int? minPts = null, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (epsilon.HasValue && epsilon.Value <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon),
                "Epsilon must be positive if specified.");
        }

        if (minPts.HasValue && minPts.Value < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(minPts),
                "MinPts must be at least 1 if specified.");
        }

        _epsilon = epsilon;
        _minPts = minPts;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        _trainingData = X;

        // Estimate parameters if not provided
        _fittedMinPts = _minPts ?? Math.Max(2, 2 * X.Columns);
        _fittedEpsilon = _epsilon ?? EstimateEpsilon(X);

        // Run DBSCAN clustering
        _clusterLabels = RunDBSCAN(X);

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
            // Count neighbors within epsilon
            int neighborCount = CountNeighborsInRadius(X, i);

            // Score based on density (fewer neighbors = higher anomaly score)
            // Core points (neighborCount >= minPts) get low scores
            // Border and noise points get higher scores
            double score;
            if (neighborCount >= _fittedMinPts)
            {
                // Core point - low anomaly score
                score = 1.0 / (neighborCount + 1);
            }
            else if (neighborCount > 0)
            {
                // Border point - medium anomaly score
                score = 1.0 - (neighborCount / (double)_fittedMinPts);
            }
            else
            {
                // Noise point - high anomaly score
                score = 1.0;
            }

            scores[i] = NumOps.FromDouble(score);
        }

        return scores;
    }

    private int[] RunDBSCAN(Matrix<T> X)
    {
        int n = X.Rows;
        var labels = new int[n];
        for (int i = 0; i < n; i++)
        {
            labels[i] = -1; // -1 = unvisited
        }

        int clusterId = 0;

        for (int i = 0; i < n; i++)
        {
            if (labels[i] != -1) continue;

            var neighbors = GetNeighborsInRadius(X, i);

            if (neighbors.Count < _fittedMinPts)
            {
                labels[i] = 0; // Noise (label 0 = noise)
            }
            else
            {
                clusterId++;
                ExpandCluster(X, i, neighbors, labels, clusterId);
            }
        }

        return labels;
    }

    private void ExpandCluster(Matrix<T> X, int pointIdx, List<int> neighbors, int[] labels, int clusterId)
    {
        labels[pointIdx] = clusterId;

        var queue = new Queue<int>(neighbors);

        while (queue.Count > 0)
        {
            int current = queue.Dequeue();

            if (labels[current] == 0) // Was noise, now border point
            {
                labels[current] = clusterId;
            }

            if (labels[current] != -1) continue; // Already processed

            labels[current] = clusterId;

            var currentNeighbors = GetNeighborsInRadius(X, current);

            if (currentNeighbors.Count >= _fittedMinPts)
            {
                foreach (int neighbor in currentNeighbors)
                {
                    if (labels[neighbor] <= 0) // Unvisited or noise
                    {
                        queue.Enqueue(neighbor);
                    }
                }
            }
        }
    }

    private double EstimateEpsilon(Matrix<T> X)
    {
        // K-distance method: find the elbow in sorted k-distances
        int k = _fittedMinPts;
        var kDistances = new List<double>();

        for (int i = 0; i < X.Rows; i++)
        {
            var distances = new List<double>();
            var pointI = new Vector<T>(X.Columns);
            for (int j = 0; j < X.Columns; j++)
            {
                pointI[j] = X[i, j];
            }

            for (int j = 0; j < X.Rows; j++)
            {
                if (i == j) continue;

                var pointJ = new Vector<T>(X.Columns);
                for (int c = 0; c < X.Columns; c++)
                {
                    pointJ[c] = X[j, c];
                }

                distances.Add(NumOps.ToDouble(StatisticsHelper<T>.EuclideanDistance(pointI, pointJ)));
            }

            distances.Sort();
            if (distances.Count >= k)
            {
                kDistances.Add(distances[k - 1]);
            }
        }

        kDistances.Sort();

        // Use the knee/elbow point or a percentile
        // Simple heuristic: use the 90th percentile k-distance
        int elbowIdx = (int)(kDistances.Count * 0.9);
        return kDistances[Math.Min(elbowIdx, kDistances.Count - 1)];
    }

    private List<int> GetNeighborsInRadius(Matrix<T> X, int pointIdx)
    {
        var neighbors = new List<int>();
        var point = new Vector<T>(X.Columns);
        for (int j = 0; j < X.Columns; j++)
        {
            point[j] = X[pointIdx, j];
        }

        var trainingData = _trainingData;
        if (trainingData == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        for (int i = 0; i < trainingData.Rows; i++)
        {
            var refPoint = new Vector<T>(trainingData.Columns);
            for (int j = 0; j < trainingData.Columns; j++)
            {
                refPoint[j] = trainingData[i, j];
            }

            double dist = NumOps.ToDouble(StatisticsHelper<T>.EuclideanDistance(point, refPoint));
            if (dist <= _fittedEpsilon)
            {
                neighbors.Add(i);
            }
        }

        return neighbors;
    }

    private int CountNeighborsInRadius(Matrix<T> X, int pointIdx)
    {
        return GetNeighborsInRadius(X, pointIdx).Count;
    }
}
