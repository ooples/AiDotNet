using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.DistanceBased;

/// <summary>
/// Detects anomalies using K-Nearest Neighbors distance.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> KNN anomaly detection identifies outliers based on their distance
/// to their k-nearest neighbors. Points that are far from their nearest neighbors are
/// considered anomalies.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. For each point, find the k nearest neighbors
/// 2. Calculate the average distance to these neighbors
/// 3. Points with large average distances are anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When anomalies are expected to be isolated from normal clusters
/// - Works well with low-to-medium dimensional data
/// - No assumption about data distribution
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - K (neighbors): 5 (typical range 3-20)
/// - Contamination: 0.1 (10%)
/// </para>
/// </remarks>
public class KNNDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _k;
    private Matrix<T>? _trainingData;
    private int _nFeatures;

    /// <summary>
    /// Gets the number of neighbors used for detection.
    /// </summary>
    public int K => _k;

    /// <summary>
    /// Creates a new KNN anomaly detector.
    /// </summary>
    /// <param name="k">Number of nearest neighbors to consider. Default is 5.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public KNNDetector(int k = 5, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (k < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(k),
                "K must be at least 1. Recommended range is 3-20.");
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

        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            // Get distances to all training points
            var distances = ComputeDistances(X, i, _trainingData!);

            // Sort and get k-nearest
            var sortedDistances = distances.ToArray()
                .Select((d, idx) => (Distance: NumOps.ToDouble(d), Index: idx))
                .OrderBy(x => x.Distance)
                .ToList();

            // Take k nearest neighbors
            // Only skip the FIRST zero-distance point as the potential self-match
            // Include subsequent zero-distance points (legitimate duplicates)
            double avgDistance = 0;
            int count = 0;
            bool skippedSelf = false;

            foreach (var (distance, _) in sortedDistances)
            {
                if (count >= _k) break;

                // Skip only the first exact self-match (distance == 0)
                if (!skippedSelf && distance == 0)
                {
                    skippedSelf = true;
                    continue;
                }

                avgDistance += distance;
                count++;
            }

            avgDistance = count > 0 ? avgDistance / count : 0;
            scores[i] = NumOps.FromDouble(avgDistance);
        }

        return scores;
    }

    private Vector<T> ComputeDistances(Matrix<T> X, int rowIndex, Matrix<T> reference)
    {
        var distances = new Vector<T>(reference.Rows);
        var queryPoint = new Vector<T>(X.Columns);

        for (int j = 0; j < X.Columns; j++)
        {
            queryPoint[j] = X[rowIndex, j];
        }

        for (int i = 0; i < reference.Rows; i++)
        {
            var refPoint = new Vector<T>(reference.Columns);
            for (int j = 0; j < reference.Columns; j++)
            {
                refPoint[j] = reference[i, j];
            }

            distances[i] = StatisticsHelper<T>.EuclideanDistance(queryPoint, refPoint);
        }

        return distances;
    }
}
