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
public partial class COFDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _k;
    [Buffer]
    private Matrix<T>? _trainingData;
    private Tensor<T>? _trainingDistanceMatrix;
    [Buffer]
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
        var data = AnomalyTensorOps.ToTensor(X);
        _trainingDistanceMatrix = AnomalyTensorOps.PairwiseDistances(data, data);
        _chainingDistances = ComputeAverageChainDistances(
            _trainingDistanceMatrix, _trainingDistanceMatrix, excludeSameIndex: true, out _).ToVector();

        // Calculate scores for training data to set threshold
        // The threshold is calibrated on leave-one-out training scores: each training point's neighbourhood
        // excludes the point itself (PyOD's decision_scores_).
        var trainingScores = ScoreAnomaliesInternal(X, scoringTrainingData: true);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Every row is scored against the whole training set, as PyOD's decision_function does: a row equal to a
    /// training point keeps that point as a zero-distance neighbour. The result depends only on the values. It
    /// used to take the leave-one-out path only when handed the very matrix instance Fit saw, so an equal copy -
    /// or a clone of the detector - scored the same data differently.
    /// </remarks>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X, scoringTrainingData: false);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X, bool scoringTrainingData)
    {
        ValidateInput(X);

        // Both fitted fields are resolved ONCE, before either branch. The two arms previously
        // repeated the same _trainingDistanceMatrix null guard, and reached the training data
        // through a null-forgiving operator that would have surfaced as a NullReferenceException
        // from inside ComputeDistanceMatrix rather than as a "not fitted" message.
        var trainingDistanceMatrix = _trainingDistanceMatrix
            ?? throw new InvalidOperationException(
                $"{nameof(COFDetector<T>)} is not fitted: the training distance matrix is missing.");
        var trainingData = _trainingData
            ?? throw new InvalidOperationException(
                $"{nameof(COFDetector<T>)} is not fitted: the training data is missing.");

        var chainingDist = _chainingDistances ?? throw new InvalidOperationException("_chainingDistances has not been initialized.");
        var engine = Engine;
        var distanceMatrix = scoringTrainingData
            ? trainingDistanceMatrix
            : AnomalyTensorOps.PairwiseDistances(AnomalyTensorOps.ToTensor(X), AnomalyTensorOps.ToTensor(trainingData));

        var acDist = ComputeAverageChainDistances(distanceMatrix, trainingDistanceMatrix, scoringTrainingData, out var neighbors);
        var avgNeighborAcDist = engine.TensorDivideScalar(
            engine.ReduceSum(AnomalyTensorOps.GatherByIndex(chainingDist.AsTensor(), neighbors), new[] { 1 }, false),
            NumOps.FromDouble(_k));

        // COF = ac-dist(point) / average(ac-dist(neighbors)), 1 when that average is not positive
        return engine.TensorWhere(
            engine.TensorGreaterThan(avgNeighborAcDist, NumOps.Zero),
            engine.TensorDivide(acDist, avgNeighborAcDist),
            Tensor<T>.CreateOnes(new[] { X.Rows })).ToVector();
    }

    // Average chaining distance of every query row over its k nearest training neighbours.
    // The chain is built greedily from the query point: at each step the remaining neighbour with the
    // shortest edge to *any* point already in the chain (the query or a visited neighbour) joins, the
    // first in nearest-neighbour order on ties; sorting once by distance to the query is not equivalent
    // and overstates the cost for curved or corridor-shaped clusters. Step s costs weight
    // 2 * (k - s) / (k * (k + 1)). All query rows advance through the k steps together.
    private Tensor<T> ComputeAverageChainDistances(Tensor<T> queryToTrainingDistances, Tensor<T> trainingDistances,
        bool excludeSameIndex, out Tensor<int> neighbors)
    {
        var engine = Engine;
        int m = queryToTrainingDistances.Shape[0];
        int nTraining = trainingDistances.Shape[0];
        int k = _k;
        T maxValue = NumOps.MaxValue;

        // k nearest neighbours (by index on ties of the stable order) and the query-to-neighbour edges
        var masked = queryToTrainingDistances;
        if (excludeSameIndex)
        {
            masked = engine.TensorMaskedFill(masked, engine.TensorEqScalar(engine.TensorEye<T>(m), NumOps.One), maxValue);
        }

        var (sorted, order) = engine.TensorSort(masked, 1, false);
        neighbors = engine.TensorSlice(order, new[] { 0, 0 }, new[] { m, k });
        var edge = engine.TensorSlice(sorted, new[] { 0, 0 }, new[] { m, k });

        var remaining = Tensor<T>.CreateOnes(new[] { m, k });
        var unavailable = engine.Fill(m * k, maxValue).AsTensor().Reshape(new[] { m, k });
        var positions = engine.TensorBroadcastTo(
            engine.Reshape(engine.TensorArange<T>(0, k, 1), new[] { 1, k }), new[] { m, k });
        var flatTraining = engine.Reshape(trainingDistances, new[] { nTraining * nTraining });
        var neighborRowOffsets = engine.TensorMultiplyScalar(neighbors, nTraining);
        var steps = new Tensor<T>[k];

        for (int s = 0; s < k; s++)
        {
            // The remaining neighbour with the shortest edge to the chain, and that edge
            var candidates = engine.TensorWhere(remaining, edge, unavailable);
            var chosen = engine.TensorArgMin(candidates, 1);
            steps[s] = engine.TensorAmin(candidates, new[] { 1 }, false);

            // It leaves the remaining set, and every other neighbour's edge to the chain can now go
            // through it: edge[i, j] = min(edge[i, j], d(neighbor_ij, chosen_i))
            var chosenColumn = engine.Reshape(chosen, new[] { m, 1 });
            var chosenMask = engine.TensorEquals(positions,
                engine.TensorBroadcastTo(engine.Reshape(chosen.Cast<T>(), new[] { m, 1 }), new[] { m, k }));
            remaining = engine.TensorSubtract(remaining, chosenMask);
            var chosenIndex = engine.TensorTakeAlongDim(neighbors, chosenColumn, 1);
            var flat = engine.TensorAdd(neighborRowOffsets, engine.TensorBroadcastTo(chosenIndex, new[] { m, k }));
            edge = engine.TensorMin(edge, AnomalyTensorOps.GatherByIndex(flatTraining, flat));
        }

        // Weighted sum of the step costs with weights 2 * (k - s) / (k * (k + 1))
        var weights = engine.TensorDivideScalar(
            engine.TensorMultiplyScalar(engine.ScalarMinusTensor(NumOps.FromDouble(k), engine.TensorArange<T>(0, k, 1)), NumOps.FromDouble(2)),
            NumOps.FromDouble((double)k * (k + 1)));
        var stepCosts = engine.TensorStack(steps, 1);
        return engine.ReduceSum(AnomalyTensorOps.MultiplyRow(stepCosts, weights), new[] { 1 }, false);
    }
}
