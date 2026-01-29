using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.DistanceBased;

/// <summary>
/// Detects anomalies using Influenced Outlierness (INFLO).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> INFLO combines the concepts of k-nearest neighbors and reverse
/// nearest neighbors. It considers not only which points are close to a given point,
/// but also which points consider the given point as their neighbor.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Find k-nearest neighbors (kNN) for each point
/// 2. Find reverse k-nearest neighbors (RkNN) - points that have this point as a neighbor
/// 3. Compute influence space = kNN union RkNN
/// 4. Compare point's density to its influence space's density
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When boundary between clusters have outliers
/// - When LOF fails at cluster boundaries
/// - Similar scenarios to LOF but with better boundary handling
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - K (neighbors): 10
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Jin, W., et al. (2006). "Mining Top-n Local Outliers in Large Databases." KDD.
/// </para>
/// </remarks>
public class INFLODetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _k;
    private Matrix<T>? _trainingData;
    private double[]? _localDensities;
    private List<int>[]? _knnLists;
    private List<int>[]? _rknnLists;
    private HashSet<int>[]? _influenceSpaces;
    private int _nFeatures;

    /// <summary>
    /// Gets the number of neighbors used for detection.
    /// </summary>
    public int K => _k;

    /// <summary>
    /// Creates a new INFLO anomaly detector.
    /// </summary>
    /// <param name="k">Number of nearest neighbors to consider. Default is 10.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public INFLODetector(int k = 10, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (k < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(k),
                "K must be at least 1. Recommended value is 10.");
        }

        _k = k;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Rows <= _k)
        {
            throw new ArgumentException(
                $"Number of samples ({X.Rows}) must be greater than k ({_k}).",
                nameof(X));
        }

        _trainingData = X;
        _nFeatures = X.Columns;

        // Build kNN and RkNN lists
        BuildNeighborLists(X);

        // Precompute influence spaces (kNN union RkNN) for each training point
        int n = X.Rows;
        _influenceSpaces = new HashSet<int>[n];
        for (int i = 0; i < n; i++)
        {
            _influenceSpaces[i] = new HashSet<int>(_knnLists![i]);
            foreach (int rknnIdx in _rknnLists![i])
            {
                _influenceSpaces[i].Add(rknnIdx);
            }
        }

        // Compute local densities
        _localDensities = ComputeLocalDensities(X);

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

        if (X.Columns != _nFeatures)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_nFeatures} features.",
                nameof(X));
        }

        var localDensities = _localDensities;
        var influenceSpaces = _influenceSpaces;
        if (localDensities == null || influenceSpaces == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            // Get k-nearest neighbors from training data to this point
            var knn = GetKNearestNeighbors(X, i);

            // For new data, use the union of the precomputed influence spaces
            // of the k-nearest training neighbors as an approximation
            var influenceSpace = new HashSet<int>(knn);
            foreach (int neighborIdx in knn)
            {
                if (neighborIdx < influenceSpaces.Length)
                {
                    foreach (int influenceIdx in influenceSpaces[neighborIdx])
                    {
                        influenceSpace.Add(influenceIdx);
                    }
                }
            }

            // Compute local density for this point using its k-nearest neighbors
            double localDensity = ComputeLocalDensityForPoint(X, i, knn);

            // Compute average local density of influence space
            double avgInfluenceDensity = 0;
            int validCount = 0;
            foreach (int neighborIdx in influenceSpace)
            {
                if (neighborIdx < localDensities.Length)
                {
                    avgInfluenceDensity += localDensities[neighborIdx];
                    validCount++;
                }
            }

            if (validCount > 0)
            {
                avgInfluenceDensity /= validCount;
            }

            // INFLO = average influence density / local density
            double inflo = localDensity > 0 ? avgInfluenceDensity / localDensity : 1.0;
            scores[i] = NumOps.FromDouble(inflo);
        }

        return scores;
    }

    private void BuildNeighborLists(Matrix<T> X)
    {
        int n = X.Rows;
        _knnLists = new List<int>[n];
        _rknnLists = new List<int>[n];

        for (int i = 0; i < n; i++)
        {
            _knnLists[i] = new List<int>();
            _rknnLists[i] = new List<int>();
        }

        // Compute distance matrix and build kNN lists
        var distances = new List<(int Index, double Distance)>[n];

        for (int i = 0; i < n; i++)
        {
            distances[i] = new List<(int, double)>();
            var pointI = new Vector<T>(X.Columns);
            for (int j = 0; j < X.Columns; j++)
            {
                pointI[j] = X[i, j];
            }

            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;

                var pointJ = new Vector<T>(X.Columns);
                for (int k = 0; k < X.Columns; k++)
                {
                    pointJ[k] = X[j, k];
                }

                double dist = NumOps.ToDouble(StatisticsHelper<T>.EuclideanDistance(pointI, pointJ));
                distances[i].Add((j, dist));
            }

            // Sort and take k nearest
            distances[i].Sort((a, b) => a.Distance.CompareTo(b.Distance));
            _knnLists[i] = distances[i].Take(_k).Select(x => x.Index).ToList();
        }

        // Build RkNN lists from kNN lists
        for (int i = 0; i < n; i++)
        {
            foreach (int neighbor in _knnLists[i])
            {
                _rknnLists[neighbor].Add(i);
            }
        }
    }

    private double[] ComputeLocalDensities(Matrix<T> X)
    {
        var densities = new double[X.Rows];

        for (int i = 0; i < X.Rows; i++)
        {
            densities[i] = ComputeLocalDensityForPoint(X, i, _knnLists![i]);
        }

        return densities;
    }

    private double ComputeLocalDensityForPoint(Matrix<T> X, int pointIdx, List<int> neighbors)
    {
        if (neighbors.Count == 0) return 0;

        var point = new Vector<T>(X.Columns);
        for (int j = 0; j < X.Columns; j++)
        {
            point[j] = X[pointIdx, j];
        }

        // Local density = 1 / (average distance to k-nearest neighbors)
        var trainingData = _trainingData;
        if (trainingData == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        double avgDist = 0;
        foreach (int neighborIdx in neighbors)
        {
            var neighbor = new Vector<T>(trainingData.Columns);
            for (int j = 0; j < trainingData.Columns; j++)
            {
                neighbor[j] = trainingData[neighborIdx, j];
            }

            avgDist += NumOps.ToDouble(StatisticsHelper<T>.EuclideanDistance(point, neighbor));
        }

        avgDist /= neighbors.Count;

        return avgDist > 0 ? 1.0 / avgDist : double.MaxValue;
    }

    private List<int> GetKNearestNeighbors(Matrix<T> X, int pointIdx)
    {
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

        return Enumerable.Range(0, trainingData.Rows)
            .Select(i =>
            {
                var refPoint = new Vector<T>(trainingData.Columns);
                for (int j = 0; j < trainingData.Columns; j++)
                {
                    refPoint[j] = trainingData[i, j];
                }
                return (Index: i, Dist: NumOps.ToDouble(StatisticsHelper<T>.EuclideanDistance(point, refPoint)));
            })
            .Where(x => x.Dist > 0)
            .OrderBy(x => x.Dist)
            .Take(_k)
            .Select(x => x.Index)
            .ToList();
    }
}
