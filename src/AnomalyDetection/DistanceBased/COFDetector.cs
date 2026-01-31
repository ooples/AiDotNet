using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.DistanceBased;

/// <summary>
/// Detects anomalies using Connectivity-based Outlier Factor (COF).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> COF improves on LOF by considering the connectivity patterns
/// of data points. It's particularly effective at detecting outliers that lie along
/// low-density paths between clusters.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Build a chaining distance based on connectivity paths
/// 2. Compare local chaining distance to neighbors' chaining distances
/// 3. Points with relatively high chaining distances are anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When outliers lie on low-density paths between clusters
/// - When LOF fails to detect certain types of outliers
/// - Data has non-uniform density
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - K (neighbors): 10
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Tang, J., et al. (2002). "Enhancing Effectiveness of Outlier Detections
/// for Low Density Patterns." PAKDD.
/// </para>
/// </remarks>
public class COFDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _k;
    private Matrix<T>? _trainingData;
    private double[]? _chainingDistances;

    /// <summary>
    /// Gets the number of neighbors used for detection.
    /// </summary>
    public int K => _k;

    /// <summary>
    /// Creates a new COF anomaly detector.
    /// </summary>
    /// <param name="k">Number of nearest neighbors to consider. Default is 10.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public COFDetector(int k = 10, double contamination = 0.1, int randomSeed = 42)
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

        // Compute chaining distances for training data
        _chainingDistances = ComputeChainingDistances(X);

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

        // Compute distance matrix
        var distanceMatrix = ComputeDistanceMatrix(X, _trainingData!);

        for (int i = 0; i < X.Rows; i++)
        {
            // Get k-nearest neighbors and their distances
            var neighbors = GetKNearestNeighbors(distanceMatrix, i, _k);

            // Compute average chaining distance (ac-dist)
            double acDist = ComputeAverageChainDist(distanceMatrix, i, neighbors);

            // Compute COF as ratio of local ac-dist to neighbors' ac-dist
            double avgNeighborAcDist = 0;
            foreach (int neighborIdx in neighbors)
            {
                avgNeighborAcDist += _chainingDistances![neighborIdx];
            }
            avgNeighborAcDist /= neighbors.Count;

            // COF = ac-dist(point) / average(ac-dist(neighbors))
            double cof = avgNeighborAcDist > 0 ? acDist / avgNeighborAcDist : 1.0;
            scores[i] = NumOps.FromDouble(cof);
        }

        return scores;
    }

    private double[] ComputeChainingDistances(Matrix<T> X)
    {
        var chainDist = new double[X.Rows];
        var distanceMatrix = ComputeDistanceMatrix(X, X);

        for (int i = 0; i < X.Rows; i++)
        {
            var neighbors = GetKNearestNeighbors(distanceMatrix, i, _k);
            chainDist[i] = ComputeAverageChainDist(distanceMatrix, i, neighbors);
        }

        return chainDist;
    }

    private double ComputeAverageChainDist(double[,] distanceMatrix, int pointIdx, List<int> neighbors)
    {
        if (neighbors.Count == 0) return 0;

        // Build SBN-path (Set-Based Nearest) chaining distance
        var visited = new HashSet<int> { pointIdx };
        double totalChainDist = 0;

        // Process neighbors in order of distance
        var orderedNeighbors = neighbors
            .OrderBy(n => distanceMatrix[pointIdx, n])
            .ToList();

        for (int i = 0; i < orderedNeighbors.Count; i++)
        {
            int current = orderedNeighbors[i];

            // Find minimum distance from current to any visited point
            double minDist = double.MaxValue;
            foreach (int v in visited)
            {
                if (distanceMatrix[current, v] < minDist)
                {
                    minDist = distanceMatrix[current, v];
                }
            }

            totalChainDist += (2.0 * (_k - i) / (_k * (_k + 1))) * minDist;
            visited.Add(current);
        }

        return totalChainDist;
    }

    private double[,] ComputeDistanceMatrix(Matrix<T> X, Matrix<T> reference)
    {
        var matrix = new double[X.Rows, reference.Rows];

        for (int i = 0; i < X.Rows; i++)
        {
            var pointI = new Vector<T>(X.Columns);
            for (int j = 0; j < X.Columns; j++)
            {
                pointI[j] = X[i, j];
            }

            for (int k = 0; k < reference.Rows; k++)
            {
                var pointK = new Vector<T>(reference.Columns);
                for (int j = 0; j < reference.Columns; j++)
                {
                    pointK[j] = reference[k, j];
                }

                matrix[i, k] = NumOps.ToDouble(StatisticsHelper<T>.EuclideanDistance(pointI, pointK));
            }
        }

        return matrix;
    }

    private List<int> GetKNearestNeighbors(double[,] distanceMatrix, int pointIdx, int k)
    {
        return Enumerable.Range(0, distanceMatrix.GetLength(1))
            .Where(i => i != pointIdx || distanceMatrix[pointIdx, i] > 0)
            .OrderBy(i => distanceMatrix[pointIdx, i])
            .Take(k)
            .ToList();
    }
}
