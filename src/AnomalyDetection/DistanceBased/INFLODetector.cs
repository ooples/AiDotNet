using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.InstanceBased)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Mining Top-n Local Outliers in Large Databases", "https://doi.org/10.1145/1150402.1150438", Year = 2006, Authors = "Wen Jin, Anthony K. H. Tung, Jiawei Han, Wei Wang")]
public partial class INFLODetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _k;
    [Buffer]
    private Matrix<T>? _trainingData;
    [Buffer]
    private Vector<T>? _localDensities;
    // [n, k] kNN of every training point (itself excluded). The influence space IS(j) = kNN(j) union RkNN(j)
    // follows from it: i is in RkNN(j) exactly when j is in kNN(i).
    private Tensor<int>? _knnIndices;
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

        // kNN of every training point (itself excluded by index) and its local density
        var engine = Engine;
        int n = X.Rows;
        var data = AnomalyTensorOps.ToTensor(X);
        var distances = AnomalyTensorOps.NearestNeighbors(data, data, _k, true, out var neighbors);
        _localDensities = LocalDensities(engine.ReduceSum(distances, new[] { 1 }, false), NumOps.FromDouble(_k)).ToVector();

        // Influence spaces kNN union RkNN are derived from the kNN lists at scoring time (no [n, n] matrix)
        _knnIndices = neighbors;

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
        var knnIndices = _knnIndices;
        var trainingData = _trainingData;
        if (localDensities == null || knnIndices == null || trainingData == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        // Query rows are scored in bounded blocks (only [block, nTraining * k] tensors are live)
        var training = AnomalyTensorOps.ToTensor(trainingData);
        int nTraining = trainingData.Rows;
        int slots = Math.Min(_k, nTraining);
        long perRow = (long)nTraining * Math.Max(1, knnIndices.Shape[1]);
        return AnomalyTensorOps.MapRowBlocks(AnomalyTensorOps.ToTensor(X), perRow,
            query => ScoreBlock(query, training, localDensities.AsTensor(), knnIndices, slots)).ToVector();
    }

    // INFLO scores of one block of query rows.
    private Tensor<T> ScoreBlock(Tensor<T> query, Tensor<T> training, Tensor<T> localDensities, Tensor<int> knnIndices,
        int slots)
    {
        var engine = Engine;
        int m = query.Shape[0];
        int nTraining = training.Shape[0];

        // k nearest training points at a positive distance (nearest first, by index on ties). Distances
        // sort ascending, so after the zero distances the next (at most) k entries are the neighbours.
        var (sorted, order) = AnomalyTensorOps.StableSortRows(
            AnomalyTensorOps.PairwiseDistances(query, training), descending: false);
        var positive = engine.TensorGreaterThan(sorted, NumOps.Zero);
        var neighborMask = engine.TensorMultiply(positive,
            engine.TensorLessThan(engine.TensorCumSum(positive, 1), NumOps.FromDouble(_k + 0.5)));
        var count = engine.ReduceSum(neighborMask, new[] { 1 }, false);

        // Local density from the neighbour distances; 0 without neighbours
        var localDensity = engine.TensorWhere(
            engine.TensorGreaterThan(count, NumOps.Zero),
            LocalDensities(engine.ReduceSum(engine.TensorMultiply(neighborMask, sorted), new[] { 1 }, false), count),
            Tensor<T>.CreateZeros(new[] { m }));

        // For new data the influence space is the kNN N(q) plus the union of the influence spaces
        // IS(j) = kNN(j) union RkNN(j) of those training neighbours j:
        // (a) i in kNN(j) for a neighbour j: scatter the kNN lists of the (at most k) neighbours, which a stable
        //     descending sort of the mask moves to the first slots;
        // (b) i in RkNN(j) for a neighbour j, i.e. kNN(i) meets N(q): gather N(q) at every kNN(i) and take the max.
        var knnMatrix = engine.TensorTakeAlongDim(neighborMask, engine.TensorArgsort(order, 1, false), 1);
        int listLength = knnIndices.Shape[1];
        var (slotMask, slotPositions) = AnomalyTensorOps.StableSortRows(neighborMask, descending: true);
        var firstSlots = engine.TensorSlice(slotPositions, new[] { 0, 0 }, new[] { m, slots });
        var neighborIndices = engine.TensorTakeAlongDim(order, firstSlots, 1);
        var valid = engine.TensorSlice(slotMask, new[] { 0, 0 }, new[] { m, slots });
        var neighborLists = engine.Reshape(
            engine.TensorGather(knnIndices, engine.Reshape(neighborIndices, new[] { m * slots }), 0),
            new[] { m, slots * listLength });
        var listWeights = engine.Reshape(engine.TensorBroadcastTo(engine.Reshape(valid, new[] { m, slots, 1 }),
            new[] { m, slots, listLength }), new[] { m, slots * listLength });
        var inNeighborKnn = engine.TensorGreaterThan(
            engine.TensorScatterAdd(Tensor<T>.CreateZeros(new[] { m, nTraining }), neighborLists, listWeights, 1), NumOps.Zero);
        var reverse = engine.ReduceMax(engine.Reshape(
            engine.TensorGather(knnMatrix, engine.Reshape(knnIndices, new[] { nTraining * listLength }), 1),
            new[] { m, nTraining, listLength }), new[] { 2 }, false);
        var membership = engine.TensorClamp(
            engine.TensorAdd(engine.TensorAdd(knnMatrix, inNeighborKnn), reverse), NumOps.Zero, NumOps.One);
        var influenceCount = engine.ReduceSum(membership, new[] { 1 }, false);
        var influenceSum = engine.Reshape(
            engine.TensorMatMul(membership, engine.Reshape(localDensities, new[] { nTraining, 1 })), new[] { m });
        var hasInfluence = engine.TensorGreaterThan(influenceCount, NumOps.Zero);
        var avgInfluenceDensity = engine.TensorWhere(hasInfluence,
            engine.TensorDivide(influenceSum, engine.TensorWhere(hasInfluence, influenceCount, Tensor<T>.CreateOnes(new[] { m }))),
            Tensor<T>.CreateZeros(new[] { m }));

        // INFLO = average influence density / local density, 1 when the local density is not positive
        return engine.TensorWhere(
            engine.TensorGreaterThan(localDensity, NumOps.Zero),
            engine.TensorDivide(avgInfluenceDensity, localDensity),
            Tensor<T>.CreateOnes(new[] { m }));
    }

    // Local density 1 / (average neighbour distance) from the distance sums and neighbour counts,
    // MaxValue when the average is 0.
    private Tensor<T> LocalDensities(Tensor<T> distanceSums, Tensor<T> counts)
    {
        var engine = Engine;
        int m = distanceSums.Length;
        var avgDist = engine.TensorDivide(distanceSums, counts);
        var positive = engine.TensorGreaterThan(avgDist, NumOps.Zero);
        return engine.TensorWhere(positive,
            engine.TensorReciprocal(engine.TensorWhere(positive, avgDist, Tensor<T>.CreateOnes(new[] { m }))),
            engine.Fill(m, NumOps.MaxValue).AsTensor());
    }

    private Tensor<T> LocalDensities(Tensor<T> distanceSums, T count)
        => LocalDensities(distanceSums, Engine.Fill(distanceSums.Length, count).AsTensor());
}
