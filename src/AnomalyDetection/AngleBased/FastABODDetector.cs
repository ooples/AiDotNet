using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.AngleBased;

/// <summary>
/// Detects anomalies using Fast Angle-Based Outlier Detection (FastABOD).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> FastABOD is an optimized version of ABOD that uses only
/// k-nearest neighbors instead of all points. This makes it much faster while
/// maintaining good detection quality.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. For each point p, find k nearest neighbors
/// 2. Compute angles only between neighbor pairs
/// 3. Calculate angle variance as outlier score
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Large datasets where full ABOD is too slow
/// - High-dimensional data
/// - When you need faster runtime with slight accuracy trade-off
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - K (neighbors): 10
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Kriegel, H., et al. (2008). "Angle-Based Outlier Detection in High-dimensional Data." KDD.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Angle-Based Outlier Detection in High-dimensional Data", "https://doi.org/10.1145/1401890.1401946", Year = 2008, Authors = "Hans-Peter Kriegel, Matthias Schubert, Arthur Zimek")]
public partial class FastABODDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _k;
    [Buffer]
    private Matrix<T>? _trainingData;

    /// <summary>
    /// Gets the number of neighbors used for detection.
    /// </summary>
    public int K => _k;

    /// <summary>
    /// Creates a new FastABOD anomaly detector.
    /// </summary>
    /// <param name="k">Number of nearest neighbors to consider. Default is 10.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public FastABODDetector(int k = 10, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (k < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(k),
                "K must be at least 2 for angle computation. Recommended is 10.");
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

        // The threshold is calibrated on leave-one-out training scores (PyOD's decision_scores_).
        var trainingScores = ScoreAnomaliesInternal(X, excludeSelf: true);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Every row is scored against the whole training set, as PyOD's decision_function does. The result depends
    /// only on the values; it used to exclude self only when handed the very matrix instance Fit saw.
    /// </remarks>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X, excludeSelf: false);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X, bool excludeSelf)
    {
        ValidateInput(X);

        var trainingData = _trainingData;
        if (trainingData == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var engine = Engine;
        int m = X.Rows;
        int nTraining = trainingData.Rows;
        int d = X.Columns;
        int k = Math.Min(_k, nTraining);
        T epsilon = NumOps.FromDouble(1e-10);
        var training = AnomalyTensorOps.ToTensor(trainingData);
        var queries = AnomalyTensorOps.ToTensor(X);

        // k nearest training points at a positive distance (zero-distance duplicates and, for the training
        // scores, the point itself skipped); nearest first, by index on ties
        var distances = AnomalyTensorOps.PairwiseDistances(queries, training);
        if (excludeSelf)
        {
            distances = engine.TensorMaskedFill(distances,
                engine.TensorEqScalar(engine.TensorEye<T>(m), NumOps.One), NumOps.Zero);
        }

        var (sorted, order) = engine.TensorSort(distances, 1, false);
        var zeroCount = engine.ReduceSum(engine.TensorEquals(sorted, Tensor<T>.CreateZeros(new[] { m, nTraining })), new[] { 1 }, false);
        var slots = AnomalyTensorOps.BroadcastRow(engine.TensorArange<T>(0, k, 1), m);
        var available = engine.TensorSubtract(engine.Fill(m, NumOps.FromDouble(nTraining)).AsTensor(), zeroCount);
        var slotValid = engine.TensorLessThan(slots, AnomalyTensorOps.BroadcastColumn(available, k));
        var positions = engine.TensorMin(engine.TensorAdd(slots, AnomalyTensorOps.BroadcastColumn(zeroCount, k)),
            NumOps.FromDouble(nTraining - 1));
        var neighborPositions = positions.Cast<int>();
        var neighborCount = engine.ReduceSum(slotValid, new[] { 1 }, false);

        // ABOF over the neighbour pairs (A, B), A before B: variance of cos(PA, PB) weighted by
        // 1 / (|PA|^2 |PB|^2), where |PA| is the neighbour distance
        var neighborIndex = engine.TensorTakeAlongDim(order, neighborPositions, 1);
        var neighborDist = engine.TensorTakeAlongDim(sorted, neighborPositions, 1);
        var neighborRows = engine.Reshape(
            engine.TensorGather(training, engine.Reshape(neighborIndex, new[] { m * k }), 0), new[] { m, k, d });
        var pa = engine.TensorSubtract(neighborRows,
            engine.TensorBroadcastTo(engine.Reshape(queries, new[] { m, 1, d }), new[] { m, k, d }));
        var dots = engine.TensorBatchMatMul(pa, engine.TensorSwapAxes(pa, 1, 2));

        var distA = engine.TensorBroadcastTo(engine.Reshape(neighborDist, new[] { m, k, 1 }), new[] { m, k, k });
        var distB = engine.TensorBroadcastTo(engine.Reshape(neighborDist, new[] { m, 1, k }), new[] { m, k, k });
        var validA = engine.TensorBroadcastTo(engine.Reshape(slotValid, new[] { m, k, 1 }), new[] { m, k, k });
        var validB = engine.TensorBroadcastTo(engine.Reshape(slotValid, new[] { m, 1, k }), new[] { m, k, k });
        var upper = engine.TensorBroadcastTo(engine.Reshape(
            engine.TensorTranspose(engine.TensorTril(Tensor<T>.CreateOnes(new[] { k, k }), -1)), new[] { 1, k, k }), new[] { m, k, k });
        var mask = engine.TensorMultiply(upper, engine.TensorMultiply(validA, validB));
        var zeros = Tensor<T>.CreateZeros(new[] { m, k, k });
        var onesCube = Tensor<T>.CreateOnes(new[] { m, k, k });

        var distProduct = engine.TensorWhere(mask, engine.TensorMultiply(distA, distB), onesCube);
        var cosAngle = engine.TensorClamp(engine.TensorDivide(dots, distProduct), NumOps.FromDouble(-1.0), NumOps.One);
        var sqProduct = engine.TensorWhere(mask,
            engine.TensorMultiply(engine.TensorMultiply(distA, distA), engine.TensorMultiply(distB, distB)), onesCube);
        var weight = engine.TensorWhere(mask, engine.TensorReciprocal(sqProduct), zeros);
        var weightedAngles = engine.TensorWhere(mask, engine.TensorMultiply(cosAngle, weight), zeros);

        var sumWeights = engine.ReduceSum(weight, new[] { 1, 2 }, false);
        var mean = engine.TensorDivide(engine.ReduceSum(weightedAngles, new[] { 1, 2 }, false), sumWeights);
        var angle = engine.TensorDivide(weightedAngles, engine.TensorWhere(mask, weight, onesCube));
        var diff = engine.TensorSubtract(angle,
            engine.TensorBroadcastTo(engine.Reshape(mean, new[] { m, 1, 1 }), new[] { m, k, k }));
        var variance = engine.ReduceSum(
            engine.TensorWhere(mask, engine.TensorMultiply(weight, engine.TensorSquare(diff)), zeros), new[] { 1, 2 }, false);

        // ABOF is MaxValue with fewer than 2 neighbours or a weight sum below epsilon; scores invert it
        var maxValue = engine.Fill(m, NumOps.MaxValue).AsTensor();
        var abof = engine.TensorWhere(
            engine.TensorMultiply(engine.TensorGreaterThan(neighborCount, NumOps.FromDouble(1.5)),
                engine.ScalarMinusTensor(NumOps.One, engine.TensorLessThan(sumWeights, epsilon))),
            engine.TensorDivide(variance, sumWeights), maxValue);
        var positive = engine.TensorGreaterThan(abof, NumOps.Zero);
        return engine.TensorWhere(positive,
            engine.TensorReciprocal(engine.TensorWhere(positive, abof, Tensor<T>.CreateOnes(new[] { m }))),
            maxValue).ToVector();
    }
}
