using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.InstanceBased)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Efficient Algorithms for Mining Outliers from Large Data Sets", "https://doi.org/10.1145/335191.335437", Year = 2000, Authors = "Sridhar Ramaswamy, Rajeev Rastogi, Kyuseok Shim")]
public partial class KNNDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _k;
    [Buffer]
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

        // The threshold is calibrated on leave-one-out training scores: each training point's
        // neighbourhood excludes the point itself (PyOD's decision_scores_).
        var trainingScores = ScoreAnomaliesInternal(X, excludeSelf: true);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Every row is scored against the whole training set, as PyOD's decision_function does: a row equal to a
    /// training point has that point as a zero-distance neighbour. The result depends only on the values. It used
    /// to exclude self only when handed the very matrix instance Fit saw, so an equal copy - or a clone of the
    /// detector - scored the same data differently.
    /// </remarks>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X, excludeSelf: false);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X, bool excludeSelf)
    {
        ValidateInput(X);

        if (X.Columns != _nFeatures)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_nFeatures} features.",
                nameof(X));
        }

        // Self-exclusion by index is only meaningful when X IS the training set, which the caller states.
        bool isSameAsTraining = excludeSelf;

        var trainingData = _trainingData ?? throw new InvalidOperationException("_trainingData is null where it is required.");
        var engine = Engine;
        int nTrain = trainingData.Rows;

        // Self-exclusion: push each row's own entry past every real distance so it sorts last.
        bool excludeOwnRow = isSameAsTraining && X.Rows == nTrain;
        int available = excludeOwnRow ? nTrain - 1 : nTrain;

        // Average of the k smallest remaining distances (the sum does not depend on how ties order),
        // reduced per bounded block of query rows so the [X.Rows, nTrain] matrix is never built.
        int count = Math.Min(_k, available);
        if (count <= 0)
        {
            return new Vector<T>(X.Rows);
        }

        var nearest = AnomalyTensorOps.NearestNeighbors(AnomalyTensorOps.ToTensor(X), AnomalyTensorOps.ToTensor(trainingData),
            count, excludeOwnRow, out _);
        return engine.TensorDivideScalar(engine.ReduceSum(nearest, new[] { 1 }, false), NumOps.FromDouble(count)).ToVector();
    }
}
