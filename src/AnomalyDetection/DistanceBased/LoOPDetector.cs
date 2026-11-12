using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.DistanceBased;

/// <summary>
/// Detects anomalies using Local Outlier Probability (LoOP).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> LoOP improves on LOF by providing a probability score between 0 and 1,
/// making results easier to interpret. A score of 0 means the point is definitely not an outlier,
/// while 1 means it's definitely an outlier.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Compute probabilistic set distance (PLOF) using Gaussian error function
/// 2. Normalize using local standard deviation of LOF values
/// 3. Convert to probability using error function
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When you need interpretable probability scores
/// - When comparing outlier scores across different datasets
/// - Similar use cases as LOF but with better interpretability
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - K (neighbors): 10
/// - Lambda: 3 (controls sensitivity)
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Kriegel, H., et al. (2009). "LoOP: Local Outlier Probabilities." CIKM.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.InstanceBased)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("LoOP: Local Outlier Probabilities", "https://doi.org/10.1145/1645953.1646195")]
public partial class LoOPDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _k;
    private readonly double _lambda;
    [Buffer]
    private Matrix<T>? _trainingData;
    [Buffer]
    private Vector<T>? _probabilisticDistances;

    /// <summary>
    /// Gets the number of neighbors used for detection.
    /// </summary>
    public int K => _k;

    /// <summary>
    /// Gets the lambda parameter (standard deviations for probability).
    /// </summary>
    public double Lambda => _lambda;

    /// <summary>
    /// Creates a new LoOP anomaly detector.
    /// </summary>
    /// <param name="k">Number of nearest neighbors to consider. Default is 10.</param>
    /// <param name="lambda">
    /// Number of standard deviations for the probability estimate. Default is 3.
    /// Higher values make the detector less sensitive.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public LoOPDetector(int k = 10, double lambda = 3.0, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (k < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(k),
                "K must be at least 1. Recommended value is 10.");
        }

        if (lambda <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(lambda),
                "Lambda must be positive. Recommended value is 3.");
        }

        _k = k;
        _lambda = lambda;
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

        // Compute probabilistic distances for training data
        _probabilisticDistances = ComputeProbabilisticDistances(X);

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

        var probDists = _probabilisticDistances ?? throw new InvalidOperationException("_probabilisticDistances has not been initialized.");
        var trainingData = _trainingData ?? throw new InvalidOperationException("Model not properly fitted.");

        var engine = Engine;
        T k = NumOps.FromDouble(_k);
        var training = AnomalyTensorOps.ToTensor(trainingData);
        var pdTraining = probDists.AsTensor();

        // PLOF of every training point within its training neighbourhood (itself excluded by index)
        AnomalyTensorOps.NearestNeighbors(training, training, _k, true, true, out var trainingNeighbors);
        var trainingExpected = engine.TensorDivideScalar(
            engine.ReduceSum(AnomalyTensorOps.GatherByIndex(pdTraining, trainingNeighbors), new[] { 1 }, false), k);
        var trainingPlof = Plof(pdTraining, trainingExpected);

        // nPLOF = lambda * sqrt(E[PLOF^2]) with the expectation over the whole training set: one aggregate
        // normalization for every object (Kriegel et al. 2009, the aggregate nPLOF; ELKI computes it once per relation)
        T nplof = NumOps.Multiply(NumOps.FromDouble(_lambda), NumOps.Sqrt(engine.TensorMean(engine.TensorSquare(trainingPlof))));

        // Query rows are scored in bounded blocks: only a [block, nTrain] distance matrix is live
        return AnomalyTensorOps.MapRowBlocks(AnomalyTensorOps.ToTensor(X), trainingData.Rows,
            query => ScoreBlock(query, training, pdTraining, nplof)).ToVector();
    }

    // LoOP scores of one block of query rows against the training data, given the dataset-wide nPLOF.
    private Tensor<T> ScoreBlock(Tensor<T> query, Tensor<T> training, Tensor<T> pdTraining, T nplof)
    {
        var engine = Engine;
        // Neighbourhood of every point: the k nearest training points at a positive distance. Distances
        // sort ascending, so after the zero distances the next (at most) k entries are the neighbours.
        var (sortedSq, order) = AnomalyTensorOps.StableSortRows(
            AnomalyTensorOps.PairwiseSquaredDistances(query, training), descending: false);
        var positive = engine.TensorGreaterThan(sortedSq, NumOps.Zero);
        var neighborMask = engine.TensorMultiply(positive,
            engine.TensorLessThan(engine.TensorCumSum(positive, 1), NumOps.FromDouble(_k + 0.5)));
        var count = engine.ReduceSum(neighborMask, new[] { 1 }, false);

        // Probabilistic distance sqrt(sum(d^2) / |S|) and its expected value over the neighbours
        var pdist = engine.TensorSqrt(engine.TensorDivide(
            engine.ReduceSum(engine.TensorMultiply(neighborMask, sortedSq), new[] { 1 }, false), count));
        var ePdist = engine.TensorDivide(
            MaskedRowSum(AnomalyTensorOps.GatherByIndex(pdTraining, order), neighborMask), count);

        // PLOF (Probabilistic LOF)
        var plof = Plof(pdist, ePdist);

        // LoOP = max(0, erf(PLOF / (sqrt(2) * nPLOF))) (Kriegel et al. 2009); with nPLOF = 0 it is 1 for a
        // positive PLOF and 0 otherwise
        if (!NumOps.GreaterThan(nplof, NumOps.Zero))
        {
            return engine.TensorGreaterThan(plof, NumOps.Zero);
        }

        var z = engine.TensorDivideScalar(plof, NumOps.Multiply(nplof, NumOps.FromDouble(Math.Sqrt(2))));
        return engine.TensorMax(engine.TensorErf(z), NumOps.Zero);
    }

    // sum_j values[i, j] over the entries where mask[i, j] is 1 (masked-out entries never contribute,
    // even when they are infinite or NaN).
    private Tensor<T> MaskedRowSum(Tensor<T> values, Tensor<T> mask)
    {
        var engine = Engine;
        var kept = engine.TensorWhere(mask, values, engine.TensorMultiplyScalar(mask, NumOps.Zero));
        return engine.ReduceSum(kept, new[] { 1 }, false);
    }

    // PLOF = pdist / E[pdist] - 1 where E[pdist] > 0, else 0.
    private Tensor<T> Plof(Tensor<T> pdist, Tensor<T> expected)
    {
        var engine = Engine;
        var ratio = engine.TensorSubtractScalar(engine.TensorDivide(pdist, expected), NumOps.One);
        return engine.TensorWhere(
            engine.TensorGreaterThan(expected, NumOps.Zero), ratio, Tensor<T>.CreateZeros(new[] { pdist.Length }));
    }

    private Vector<T> ComputeProbabilisticDistances(Matrix<T> X)
    {
        // Probabilistic set distance of every training point over its k training neighbours (itself
        // excluded by index): sqrt(sum(d^2) / k)
        var data = AnomalyTensorOps.ToTensor(X);
        var squared = AnomalyTensorOps.NearestNeighbors(data, data, _k, true, true, out _);
        var engine = Engine;
        return engine.TensorSqrt(engine.TensorDivideScalar(
            engine.ReduceSum(squared, new[] { 1 }, false), NumOps.FromDouble(_k))).ToVector();
    }
}
