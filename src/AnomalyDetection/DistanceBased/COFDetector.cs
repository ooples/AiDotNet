using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.DistanceBased;

/// <summary>
/// Detects anomalies using Connectivity-Based Outlier Factor (COF).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> COF improves on LOF by considering the connectivity pattern
/// of a point's neighborhood. It detects outliers that are connected differently
/// from their neighbors (e.g., points in low-density corridors).
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
public class COFDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _k;
    private Matrix<T>? _trainingData;
    private Vector<T>? _chainingDistances;

    /// <summary>
    /// Gets the number of neighbors used for detection.
    /// </summary>
    public int K => _k;

    /// <summary>
    /// Creates a new COF anomaly detector.
    /// </summary>
    /// <param name="k">Number of nearest neighbors. Default is 20.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public COFDetector(int k = 20, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (k < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(k),
                "K must be at least 1. Recommended is 20.");
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

        // Precompute chaining distances for training data
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
        var distanceMatrix = ComputeDistanceMatrix(X, _trainingData!);

        for (int i = 0; i < X.Rows; i++)
        {
            var neighbors = GetKNearestNeighbors(distanceMatrix, i, _k);

            T acDist = ComputeAverageChainDist(distanceMatrix, i, neighbors);

            T avgNeighborAcDist = NumOps.Zero;
            var chainingDist = _chainingDistances ?? throw new InvalidOperationException("_chainingDistances has not been initialized.");
            foreach (int neighborIdx in neighbors)
            {
                avgNeighborAcDist = NumOps.Add(avgNeighborAcDist, chainingDist[neighborIdx]);
            }
            avgNeighborAcDist = NumOps.Divide(avgNeighborAcDist, NumOps.FromDouble(neighbors.Count));

            // COF = ac-dist(point) / average(ac-dist(neighbors))
            T cof = NumOps.GreaterThan(avgNeighborAcDist, NumOps.Zero)
                ? NumOps.Divide(acDist, avgNeighborAcDist)
                : NumOps.One;
            scores[i] = cof;
        }

        return scores;
    }

    private Vector<T> ComputeChainingDistances(Matrix<T> X)
    {
        var chainDist = new Vector<T>(X.Rows);
        var distanceMatrix = ComputeDistanceMatrix(X, X);

        for (int i = 0; i < X.Rows; i++)
        {
            var neighbors = GetKNearestNeighbors(distanceMatrix, i, _k);
            chainDist[i] = ComputeAverageChainDist(distanceMatrix, i, neighbors);
        }

        return chainDist;
    }

    private T ComputeAverageChainDist(T[,] distanceMatrix, int pointIdx, List<int> neighbors)
    {
        if (neighbors.Count == 0) return NumOps.Zero;

        var visited = new HashSet<int> { pointIdx };
        T totalChainDist = NumOps.Zero;

        var orderedNeighbors = neighbors
            .OrderBy(n => NumOps.ToDouble(distanceMatrix[pointIdx, n]))
            .ToList();

        for (int i = 0; i < orderedNeighbors.Count; i++)
        {
            int current = orderedNeighbors[i];

            T minDist = NumOps.MaxValue;
            foreach (int v in visited)
            {
                if (NumOps.LessThan(distanceMatrix[current, v], minDist))
                {
                    minDist = distanceMatrix[current, v];
                }
            }

            // Weight: 2 * (k - i) / (k * (k + 1))
            T weight = NumOps.Divide(
                NumOps.FromDouble(2.0 * (_k - i)),
                NumOps.FromDouble((double)_k * (_k + 1)));
            totalChainDist = NumOps.Add(totalChainDist, NumOps.Multiply(weight, minDist));
            visited.Add(current);
        }

        return totalChainDist;
    }

    private T[,] ComputeDistanceMatrix(Matrix<T> X, Matrix<T> reference)
    {
        var matrix = new T[X.Rows, reference.Rows];

        for (int i = 0; i < X.Rows; i++)
        {
            var pointI = new Vector<T>(X.GetRowReadOnlySpan(i).ToArray());

            for (int k = 0; k < reference.Rows; k++)
            {
                var pointK = new Vector<T>(reference.GetRowReadOnlySpan(k).ToArray());

                // Vectorized Euclidean distance via Engine
                var diff = Engine.Subtract(pointI, pointK);
                matrix[i, k] = NumOps.Sqrt(Engine.DotProduct(diff, diff));
            }
        }

        return matrix;
    }

    private List<int> GetKNearestNeighbors(T[,] distanceMatrix, int pointIdx, int k)
    {
        return Enumerable.Range(0, distanceMatrix.GetLength(1))
            .Where(i => i != pointIdx || NumOps.GreaterThan(distanceMatrix[pointIdx, i], NumOps.Zero))
            .OrderBy(i => NumOps.ToDouble(distanceMatrix[pointIdx, i]))
            .Take(k)
            .ToList();
    }
}
