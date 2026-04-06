using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.ClusterBased;

/// <summary>
/// Detects anomalies using DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DBSCAN groups together points that are closely packed and
/// marks points in low-density regions as outliers (noise).
/// </para>
/// <para>
/// The algorithm works by:
/// 1. For each point, find all neighbors within epsilon distance
/// 2. If a point has at least minPts neighbors, it's a core point
/// 3. Expand clusters from core points
/// 4. Points not belonging to any cluster are noise (anomalies)
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When anomalies are in low-density regions
/// - When clusters have arbitrary shapes
/// - No need to specify number of clusters in advance
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Epsilon: estimated from data (k-distance method)
/// - MinPts: 2 * dimensions
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Ester, M., et al. (1996). "A Density-Based Algorithm for Discovering Clusters." KDD.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Clustering)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ModelPaper("A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise", "https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf", Year = 1996, Authors = "Martin Ester, Hans-Peter Kriegel, Joerg Sander, Xiaowei Xu")]
public class DBSCANDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double? _epsilon;
    private readonly int? _minPts;
    private T _fittedEpsilon;
    private int _fittedMinPts;
    private int[]? _clusterLabels;
    private Matrix<T>? _trainingData;

    /// <summary>
    /// Gets the epsilon (neighborhood radius) parameter.
    /// </summary>
    public T Epsilon => _fittedEpsilon;

    /// <summary>
    /// Gets the minimum points parameter.
    /// </summary>
    public int MinPts => _fittedMinPts;

    /// <summary>
    /// Creates a new DBSCAN anomaly detector.
    /// </summary>
    /// <param name="epsilon">Neighborhood radius. If null, estimated automatically from the data.</param>
    /// <param name="minPts">Minimum number of points to form a dense region. If null, uses 2 * dimensions.</param>
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
        _fittedEpsilon = NumOps.Zero;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        _trainingData = X;

        // Estimate parameters if not provided
        _fittedMinPts = _minPts ?? Math.Max(2, 2 * X.Columns);
        _fittedEpsilon = _epsilon.HasValue
            ? NumOps.FromDouble(_epsilon.Value)
            : EstimateEpsilon(X);

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
            int neighborCount = CountNeighborsInRadius(X, i);

            // Score based on density — all in T
            T score;
            if (neighborCount >= _fittedMinPts)
            {
                // Core point — low anomaly score
                score = NumOps.Divide(NumOps.One, NumOps.FromDouble(neighborCount + 1));
            }
            else if (neighborCount > 0)
            {
                // Border point — medium anomaly score
                score = NumOps.Subtract(NumOps.One,
                    NumOps.Divide(NumOps.FromDouble(neighborCount), NumOps.FromDouble(_fittedMinPts)));
            }
            else
            {
                // Noise point — high anomaly score
                score = NumOps.One;
            }

            scores[i] = score;
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
                labels[i] = 0; // Noise
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

            if (labels[current] == 0)
            {
                labels[current] = clusterId;
            }

            if (labels[current] != -1) continue;

            labels[current] = clusterId;

            var currentNeighbors = GetNeighborsInRadius(X, current);

            if (currentNeighbors.Count >= _fittedMinPts)
            {
                foreach (int neighbor in currentNeighbors)
                {
                    if (labels[neighbor] <= 0)
                    {
                        queue.Enqueue(neighbor);
                    }
                }
            }
        }
    }

    private T EstimateEpsilon(Matrix<T> X)
    {
        // K-distance method: find the elbow in sorted k-distances
        int k = _fittedMinPts;
        var kDistances = new List<T>();

        for (int i = 0; i < X.Rows; i++)
        {
            var distances = new List<T>();
            // Extract row as Vector for vectorized distance computation
            var pointI = new Vector<T>(X.GetRowReadOnlySpan(i).ToArray());

            for (int j = 0; j < X.Rows; j++)
            {
                if (i == j) continue;

                var pointJ = new Vector<T>(X.GetRowReadOnlySpan(j).ToArray());

                // Vectorized Euclidean distance via Engine
                var diff = Engine.Subtract(pointI, pointJ);
                T dist = NumOps.Sqrt(Engine.DotProduct(diff, diff));
                distances.Add(dist);
            }

            distances.Sort((a, b) => NumOps.Compare(a, b));
            if (distances.Count >= k)
            {
                kDistances.Add(distances[k - 1]);
            }
        }

        kDistances.Sort((a, b) => NumOps.Compare(a, b));

        // Use the 90th percentile k-distance as epsilon estimate
        int elbowIdx = (int)(kDistances.Count * 0.9);
        return kDistances[Math.Min(elbowIdx, kDistances.Count - 1)];
    }

    private List<int> GetNeighborsInRadius(Matrix<T> X, int pointIdx)
    {
        var neighbors = new List<int>();
        var point = new Vector<T>(X.GetRowReadOnlySpan(pointIdx).ToArray());

        var trainingData = _trainingData;
        if (trainingData == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        for (int i = 0; i < trainingData.Rows; i++)
        {
            var refPoint = new Vector<T>(trainingData.GetRowReadOnlySpan(i).ToArray());

            // Vectorized distance via Engine
            var diff = Engine.Subtract(point, refPoint);
            T dist = NumOps.Sqrt(Engine.DotProduct(diff, diff));

            if (!NumOps.GreaterThan(dist, _fittedEpsilon))
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
