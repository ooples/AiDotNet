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
    [ResearchPaper("Connectivity-Based Outlier Factor", "https://dl.acm.org/doi/10.1145/775047.775149")]
public class COFDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _k;
    private Matrix<T>? _trainingData;
    private T[,]? _trainingDistanceMatrix;
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

        // COF needs both query-to-training distances and distances among the training
        // neighbors themselves. Cache the square training matrix once so scoring a new,
        // differently-sized query set never indexes the rectangular query matrix as if
        // both of its axes referred to training samples.
        _trainingDistanceMatrix = ComputeDistanceMatrix(X, X);
        _chainingDistances = ComputeChainingDistances(_trainingDistanceMatrix);

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
        bool scoringTrainingData = ReferenceEquals(X, _trainingData);
        var distanceMatrix = scoringTrainingData
            ? _trainingDistanceMatrix ?? throw new InvalidOperationException("_trainingDistanceMatrix has not been initialized.")
            : ComputeDistanceMatrix(X, _trainingData!);
        var trainingDistanceMatrix = _trainingDistanceMatrix
            ?? throw new InvalidOperationException("_trainingDistanceMatrix has not been initialized.");

        for (int i = 0; i < X.Rows; i++)
        {
            var neighbors = GetKNearestNeighbors(distanceMatrix, i, _k, scoringTrainingData);

            T acDist = ComputeAverageChainDist(distanceMatrix, trainingDistanceMatrix, i, neighbors);

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

    private Vector<T> ComputeChainingDistances(T[,] distanceMatrix)
    {
        int sampleCount = distanceMatrix.GetLength(0);
        var chainDist = new Vector<T>(sampleCount);

        for (int i = 0; i < sampleCount; i++)
        {
            var neighbors = GetKNearestNeighbors(distanceMatrix, i, _k, excludeSameIndex: true);
            chainDist[i] = ComputeAverageChainDist(distanceMatrix, distanceMatrix, i, neighbors);
        }

        return chainDist;
    }

    private T ComputeAverageChainDist(
        T[,] queryToTrainingDistances,
        T[,] trainingDistances,
        int queryIndex,
        List<int> neighbors)
    {
        if (neighbors.Count == 0) return NumOps.Zero;

        var visitedNeighbors = new List<int>(neighbors.Count);
        // Preserve nearest-neighbour order for deterministic tie-breaking.
        var remainingNeighbors = new List<int>(neighbors);
        T totalChainDist = NumOps.Zero;

        // Build the minimum-cost chaining set from the query point. At each step COF
        // selects the remaining neighbour with the shortest edge to *any* point
        // already in the chain; sorting once by distance to the query is not
        // equivalent and overstates the cost for curved or corridor-shaped clusters.
        for (int i = 0; i < neighbors.Count; i++)
        {
            int current = -1;
            T minDist = NumOps.Zero;
            bool found = false;
            foreach (int candidate in remainingNeighbors)
            {
                // The query-to-neighbour edge comes from the rectangular matrix.
                // Neighbour-to-neighbour edges come from the cached square matrix.
                T candidateDistance = queryToTrainingDistances[queryIndex, candidate];
                foreach (int visited in visitedNeighbors)
                {
                    if (NumOps.LessThan(trainingDistances[candidate, visited], candidateDistance))
                    {
                        candidateDistance = trainingDistances[candidate, visited];
                    }
                }

                if (!found || NumOps.LessThan(candidateDistance, minDist))
                {
                    current = candidate;
                    minDist = candidateDistance;
                    found = true;
                }
            }

            if (!found)
            {
                break;
            }

            // Weight: 2 * (k - i) / (k * (k + 1))
            int neighborCount = neighbors.Count;
            T weight = NumOps.Divide(
                NumOps.FromDouble(2.0 * (neighborCount - i)),
                NumOps.FromDouble((double)neighborCount * (neighborCount + 1)));
            totalChainDist = NumOps.Add(totalChainDist, NumOps.Multiply(weight, minDist));
            visitedNeighbors.Add(current);
            remainingNeighbors.Remove(current);
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

    private List<int> GetKNearestNeighbors(T[,] distanceMatrix, int pointIdx, int k, bool excludeSameIndex)
    {
        return Enumerable.Range(0, distanceMatrix.GetLength(1))
            .Where(i => !excludeSameIndex || i != pointIdx)
            .OrderBy(i => NumOps.ToDouble(distanceMatrix[pointIdx, i]))
            .Take(k)
            .ToList();
    }
}
