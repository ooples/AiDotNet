using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.OutlierRemoval;

/// <summary>
/// Implements the Local Outlier Factor (LOF) algorithm for density-based outlier detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> LOF detects outliers by comparing the local density of a point
/// to the local densities of its neighbors. Points in low-density regions (relative to
/// their neighbors) are considered outliers.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Finding the k-nearest neighbors for each point
/// 2. Computing the local reachability density (LRD) for each point
/// 3. Comparing each point's LRD to its neighbors' LRDs to get the LOF score
/// 4. Points with LOF score significantly greater than 1 are outliers
/// </para>
/// <para>
/// <b>When to use:</b> LOF is particularly effective for:
/// - Detecting local outliers (points that are outliers relative to their neighborhood)
/// - Data with varying densities across different regions
/// - When you want interpretable density-based detection
/// </para>
/// <para>
/// Reference: Breunig, M. M., Kriegel, H. P., Ng, R. T., and Sander, J. (2000).
/// "LOF: Identifying Density-Based Local Outliers." ACM SIGMOD Record.
/// </para>
/// </remarks>
public class LocalOutlierFactorDetector<T> : OutlierDetectorBase<T>
{
    private readonly int _numNeighbors;
    private Matrix<T>? _trainingData;
    private Vector<T>[]? _kDistances;
    private int[][]? _neighborIndices;
    private Vector<T>? _lrd;

    /// <summary>
    /// Creates a new Local Outlier Factor detector.
    /// </summary>
    /// <param name="numNeighbors">
    /// The number of neighbors to use for computing LOF. Default is 20.
    /// Larger values provide more stable results but may miss local anomalies.
    /// Smaller values are more sensitive to local density variations.
    /// The original paper suggests values between 10 and 20 work well.
    /// </param>
    /// <param name="contamination">
    /// The expected proportion of outliers in the data. Default is 0.1 (10%).
    /// Used to set the decision threshold after fitting.
    /// </param>
    /// <param name="randomSeed">
    /// Random seed for reproducibility. Default is 42.
    /// </param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The number of neighbors (k) is the main parameter to tune:
    /// - Use smaller k (5-10) if you expect very localized anomalies
    /// - Use larger k (20-50) for more global anomaly detection
    /// - If unsure, start with k=20 and adjust based on results
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when numNeighbors is less than 1.
    /// </exception>
    public LocalOutlierFactorDetector(
        int numNeighbors = 20,
        double contamination = 0.1,
        int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (numNeighbors < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numNeighbors),
                "Number of neighbors must be at least 1.");
        }

        _numNeighbors = numNeighbors;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Rows <= _numNeighbors)
        {
            throw new ArgumentException(
                $"Number of samples ({X.Rows}) must be greater than number of neighbors ({_numNeighbors}).",
                nameof(X));
        }

        _trainingData = X;
        int n = X.Rows;
        int k = _numNeighbors;

        // Find k-nearest neighbors and k-distances for each point
        _neighborIndices = new int[n][];
        _kDistances = new Vector<T>[n];

        for (int i = 0; i < n; i++)
        {
            var (neighbors, distances) = FindKNearestNeighbors(X, i, k);
            _neighborIndices[i] = neighbors;
            _kDistances[i] = distances;
        }

        // Compute Local Reachability Density (LRD) for each training point
        _lrd = ComputeLRD(X, _neighborIndices, _kDistances);

        // Calculate LOF scores for training data to set threshold
        var trainingScores = DecisionFunctionInternal(X, true);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    /// <inheritdoc/>
    public override Vector<T> DecisionFunction(Matrix<T> X)
    {
        EnsureFitted();
        return DecisionFunctionInternal(X, false);
    }

    private Vector<T> DecisionFunctionInternal(Matrix<T> X, bool isTrainingData)
    {
        ValidateInput(X);

        int n = X.Rows;
        var scores = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            double lofScore;

            if (isTrainingData)
            {
                // Use precomputed values for training data
                lofScore = ComputeLOFScore(i);
            }
            else
            {
                // Compute LOF for new data point
                lofScore = ComputeLOFScoreForNewPoint(X.GetRow(i));
            }

            // Negate so lower values = outliers (consistent with our Predict method)
            // LOF > 1 indicates outlier, so we use -LOF
            scores[i] = NumOps.FromDouble(-lofScore);
        }

        return scores;
    }

    private (int[] neighbors, Vector<T> distances) FindKNearestNeighbors(Matrix<T> X, int pointIndex, int k)
    {
        var point = X.GetRow(pointIndex);
        var distancesWithIndices = new List<(T distance, int index)>();

        for (int j = 0; j < X.Rows; j++)
        {
            if (j != pointIndex)
            {
                T dist = EuclideanDistance(point, X, j);
                distancesWithIndices.Add((dist, j));
            }
        }

        // Sort by distance
        distancesWithIndices.Sort((a, b) => NumOps.LessThan(a.distance, b.distance) ? -1 :
            (NumOps.GreaterThan(a.distance, b.distance) ? 1 : 0));

        // Take k nearest
        var neighbors = new int[k];
        var distances = new Vector<T>(k);

        for (int i = 0; i < k; i++)
        {
            neighbors[i] = distancesWithIndices[i].index;
            distances[i] = distancesWithIndices[i].distance;
        }

        return (neighbors, distances);
    }

    private (int[] neighbors, Vector<T> distances) FindKNearestNeighborsInTraining(Vector<T> point, int k)
    {
        var distancesWithIndices = new List<(T distance, int index)>();

        for (int j = 0; j < _trainingData!.Rows; j++)
        {
            T dist = EuclideanDistance(point, _trainingData, j);
            distancesWithIndices.Add((dist, j));
        }

        // Sort by distance
        distancesWithIndices.Sort((a, b) => NumOps.LessThan(a.distance, b.distance) ? -1 :
            (NumOps.GreaterThan(a.distance, b.distance) ? 1 : 0));

        // Take k nearest
        var neighbors = new int[k];
        var distances = new Vector<T>(k);

        for (int i = 0; i < k; i++)
        {
            neighbors[i] = distancesWithIndices[i].index;
            distances[i] = distancesWithIndices[i].distance;
        }

        return (neighbors, distances);
    }

    private Vector<T> ComputeLRD(Matrix<T> X, int[][] neighborIndices, Vector<T>[] kDistances)
    {
        int n = X.Rows;
        var lrd = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            T reachabilitySum = NumOps.Zero;

            for (int j = 0; j < _numNeighbors; j++)
            {
                int neighborIdx = neighborIndices[i][j];
                T kDistNeighbor = kDistances[neighborIdx][_numNeighbors - 1]; // k-distance of neighbor
                T distToNeighbor = kDistances[i][j]; // distance to this neighbor

                // Reachability distance = max(k-distance(neighbor), distance(point, neighbor))
                T reachDist = NumOps.GreaterThan(kDistNeighbor, distToNeighbor) ? kDistNeighbor : distToNeighbor;
                reachabilitySum = NumOps.Add(reachabilitySum, reachDist);
            }

            // LRD = k / sum of reachability distances
            if (NumOps.GreaterThan(reachabilitySum, NumOps.Zero))
            {
                lrd[i] = NumOps.Divide(NumOps.FromDouble(_numNeighbors), reachabilitySum);
            }
            else
            {
                // If all neighbors are at distance 0, assign high LRD
                lrd[i] = NumOps.FromDouble(double.MaxValue);
            }
        }

        return lrd;
    }

    private double ComputeLOFScore(int pointIndex)
    {
        T lofSum = NumOps.Zero;
        var neighbors = _neighborIndices![pointIndex];
        T pointLrd = _lrd![pointIndex];

        for (int j = 0; j < _numNeighbors; j++)
        {
            int neighborIdx = neighbors[j];
            T neighborLrd = _lrd[neighborIdx];

            // LOF contribution from this neighbor
            if (NumOps.GreaterThan(pointLrd, NumOps.Zero))
            {
                lofSum = NumOps.Add(lofSum, NumOps.Divide(neighborLrd, pointLrd));
            }
        }

        // LOF = average of ratios
        return NumOps.ToDouble(lofSum) / _numNeighbors;
    }

    private double ComputeLOFScoreForNewPoint(Vector<T> point)
    {
        // Find k-nearest neighbors in training data
        var (neighbors, distances) = FindKNearestNeighborsInTraining(point, _numNeighbors);

        // Compute reachability distances and LRD for the new point
        T reachabilitySum = NumOps.Zero;

        for (int j = 0; j < _numNeighbors; j++)
        {
            int neighborIdx = neighbors[j];
            T kDistNeighbor = _kDistances![neighborIdx][_numNeighbors - 1];
            T distToNeighbor = distances[j];

            T reachDist = NumOps.GreaterThan(kDistNeighbor, distToNeighbor) ? kDistNeighbor : distToNeighbor;
            reachabilitySum = NumOps.Add(reachabilitySum, reachDist);
        }

        T pointLrd;
        if (NumOps.GreaterThan(reachabilitySum, NumOps.Zero))
        {
            pointLrd = NumOps.Divide(NumOps.FromDouble(_numNeighbors), reachabilitySum);
        }
        else
        {
            pointLrd = NumOps.FromDouble(double.MaxValue);
        }

        // Compute LOF
        T lofSum = NumOps.Zero;
        for (int j = 0; j < _numNeighbors; j++)
        {
            int neighborIdx = neighbors[j];
            T neighborLrd = _lrd![neighborIdx];

            if (NumOps.GreaterThan(pointLrd, NumOps.Zero))
            {
                lofSum = NumOps.Add(lofSum, NumOps.Divide(neighborLrd, pointLrd));
            }
        }

        return NumOps.ToDouble(lofSum) / _numNeighbors;
    }
}
