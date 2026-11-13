using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.DistanceBased;

/// <summary>
/// Detects anomalies using Local Correlation Integral (LOCI).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> LOCI is a density-based outlier detection method that automatically
/// determines the appropriate neighborhood size. It computes a Multi-Granularity Deviation
/// Factor (MDEF) that indicates how much a point's local density deviates from its
/// neighborhood's density.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. For multiple radius values, compute local density
/// 2. Compare point's density to average density of neighbors (MDEF)
/// 3. Flag points with MDEF exceeding a threshold based on standard deviation
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When you don't want to manually tune neighborhood size
/// - Data has varying local densities
/// - You want automatic threshold determination
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Alpha: 0.5 (neighborhood multiplier)
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Papadimitriou, S., et al. (2003). "LOCI: Fast Outlier Detection Using
/// the Local Correlation Integral." ICDE.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.InstanceBased)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("LOCI: Fast Outlier Detection Using the Local Correlation Integral", "https://doi.org/10.1109/ICDE.2003.1260802", Year = 2003, Authors = "Spiros Papadimitriou, Hiroyuki Kitagawa, Phillip B. Gibbons, Christos Faloutsos")]
public partial class LOCIDetector<T> : AnomalyDetectorBase<T>
{
    /// <summary>Number of radius steps used to sweep from zero to <c>_maxRadius</c>.</summary>
    private const int NumRadiiSteps = 20;

    /// <summary>Multiplier applied to <c>_maxRadius</c> so the sweep slightly exceeds it, avoiding floating-point edge cases.</summary>
    private const double RadiusOvershootFactor = 1.1;

    /// <summary>MDEF score assigned to points that have counting neighbors but zero sampling neighbors, indicating extreme isolation.</summary>
    private const double IsolatedPointScore = 1.0;

    /// <summary>Score for points with no neighbors at any radius. Chosen to be within Half max (~65504).</summary>
    private const double NoNeighborScore = 60000.0;

    private readonly double _alpha;
    private readonly int _kMax;
    [Buffer]
    private Matrix<T>? _trainingData;

    private T _maxRadius;


    /// <summary>
    /// Gets the alpha parameter (sampling neighborhood ratio).
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Gets the maximum number of neighbors to consider.
    /// </summary>
    public int KMax => _kMax;

    /// <summary>
    /// Creates a new LOCI anomaly detector.
    /// </summary>
    /// <param name="alpha">
    /// Neighborhood multiplier. The sampling neighborhood radius is alpha times the counting radius.
    /// Default is 0.5. Must be between 0 and 1.
    /// </param>
    /// <param name="kMax">Maximum number of neighbors to consider. Default is 20.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public LOCIDetector(double alpha = 0.5, int kMax = 20, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        _maxRadius = NumOps.Zero;
        if (alpha <= 0 || alpha > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(alpha),
                "Alpha must be between 0 (exclusive) and 1 (inclusive). Recommended value is 0.5.");
        }

        if (kMax < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(kMax),
                "KMax must be at least 1. Recommended value is 20.");
        }

        _alpha = alpha;
        _kMax = kMax;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Rows < 3)
        {
            throw new ArgumentException(
                "LOCI requires at least 3 samples.",
                nameof(X));
        }

        _trainingData = X;

        // Estimate max radius from data
        _maxRadius = EstimateMaxRadius(X);

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

        var trainingData = _trainingData ?? throw new InvalidOperationException("Model not properly fitted.");
        var engine = Engine;
        int m = X.Rows;
        int nTraining = trainingData.Rows;
        var training = AnomalyTensorOps.ToTensor(trainingData);

        // Radii r = maxRadius * overshoot * step / steps (slightly beyond maxRadius to handle floating-point
        // edge cases) and the sampling radii alpha * r
        var radii = engine.TensorDivideScalar(
            engine.TensorMultiplyScalar(engine.TensorArange<T>(1, NumRadiiSteps + 1, 1),
                NumOps.Multiply(_maxRadius, NumOps.FromDouble(RadiusOvershootFactor))),
            NumOps.FromDouble(NumRadiiSteps));
        var samplingRadii = engine.TensorMultiplyScalar(radii, NumOps.FromDouble(_alpha));

        // Counting neighbourhood sizes n(q, r) of every training point for every radius, from bounded
        // blocks of training rows (only a [block, nTraining] distance matrix is live)
        var trainingParts = new List<Tensor<T>>[NumRadiiSteps];
        for (int ri = 0; ri < NumRadiiSteps; ri++)
        {
            trainingParts[ri] = new List<Tensor<T>>();
        }

        int rowsPerBlock = Math.Max(1, Math.Min(nTraining, LociBlockElements / Math.Max(1, nTraining)));
        for (int first = 0; first < nTraining; first += rowsPerBlock)
        {
            int count = Math.Min(rowsPerBlock, nTraining - first);
            var blockDistances = AnomalyTensorOps.PairwiseDistances(
                engine.TensorSlice(training, new[] { first, 0 }, new[] { count, training.Shape[1] }), training);
            for (int ri = 0; ri < NumRadiiSteps; ri++)
            {
                trainingParts[ri].Add(engine.ReduceSum(
                    WithinRadius(blockDistances, engine.TensorSlice(radii, new[] { ri }, new[] { 1 })), new[] { 1 }, false));
            }
        }

        var trainingNRs = new Tensor<T>[NumRadiiSteps];
        for (int ri = 0; ri < NumRadiiSteps; ri++)
        {
            trainingNRs[ri] = trainingParts[ri].Count == 1
                ? trainingParts[ri][0]
                : engine.TensorConcatenate(trainingParts[ri].ToArray(), 0);
        }

        // Query rows are scored in bounded blocks: only a [block, nTraining] distance matrix is live
        return AnomalyTensorOps.MapRowBlocks(AnomalyTensorOps.ToTensor(X), nTraining,
            query => ScoreBlock(query, training, trainingNRs, radii, samplingRadii)).ToVector();
    }

    // Upper bound on the elements of one [rows, nTraining] block of distances.
    private const int LociBlockElements = 1 << 22;

    // LOCI scores of one block of query rows, given n(q, r) of every training point for every radius.
    private Tensor<T> ScoreBlock(Tensor<T> query, Tensor<T> training, Tensor<T>[] trainingNRs, Tensor<T> radii,
        Tensor<T> samplingRadii)
    {
        var engine = Engine;
        int m = query.Shape[0];
        int nTraining = training.Shape[0];
        var queryDistances = AnomalyTensorOps.PairwiseDistances(query, training);
        var positiveDistance = engine.TensorGreaterThan(queryDistances, NumOps.Zero);
        var zeros = Tensor<T>.CreateZeros(new[] { m });
        var ones = Tensor<T>.CreateOnes(new[] { m });
        T isolatedScore = NumOps.FromDouble(IsolatedPointScore);
        T countedFloor = NumOps.FromDouble(1.5);

        var maxMdef = zeros;
        var hadNeighbors = zeros;
        for (int ri = 0; ri < NumRadiiSteps; ri++)
        {
            // Counting neighbourhood sizes n(p, r) of the points and of every training point
            var r = engine.TensorSlice(radii, new[] { ri }, new[] { 1 });
            var nR = engine.ReduceSum(WithinRadius(queryDistances, r), new[] { 1 }, false);
            var trainingNR = trainingNRs[ri];
            var counted = engine.TensorGreaterThan(nR, countedFloor);
            hadNeighbors = engine.TensorMax(hadNeighbors, counted);

            // Sampling neighbourhood: training points at a positive distance within alpha * r
            var sampling = engine.TensorMultiply(positiveDistance,
                WithinRadius(queryDistances, engine.TensorSlice(samplingRadii, new[] { ri }, new[] { 1 })));
            var samplingCount = engine.ReduceSum(sampling, new[] { 1 }, false);
            var hasSampling = engine.TensorGreaterThan(samplingCount, NumOps.Zero);
            var safeCount = engine.TensorWhere(hasSampling, samplingCount, ones);

            // Mean and (two-pass, population) variance of n(r) over the sampling neighbourhood
            var neighborNR = AnomalyTensorOps.BroadcastRow(trainingNR, m);
            var avgNR = engine.TensorDivide(
                engine.ReduceSum(engine.TensorMultiply(sampling, neighborNR), new[] { 1 }, false), safeCount);
            var deviation = engine.TensorSubtract(neighborNR, AnomalyTensorOps.BroadcastColumn(avgNR, nTraining));
            var varNR = engine.TensorDivide(
                engine.ReduceSum(engine.TensorMultiply(sampling, engine.TensorSquare(deviation)), new[] { 1 }, false),
                safeCount);

            // MDEF = 1 - n(p,r) / avg_n(r), sigma-MDEF = std(n(r)) / avg_n(r) (both 0 when avg_n(r) <= 0);
            // score = |MDEF| / sigma-MDEF, or |MDEF| when sigma-MDEF is 0
            var positiveAvg = engine.TensorGreaterThan(avgNR, NumOps.Zero);
            var safeAvg = engine.TensorWhere(positiveAvg, avgNR, ones);
            var mdef = engine.TensorWhere(positiveAvg,
                engine.ScalarMinusTensor(NumOps.One, engine.TensorDivide(nR, safeAvg)), zeros);
            var sigmaMdef = engine.TensorWhere(positiveAvg, engine.TensorDivide(engine.TensorSqrt(varNR), safeAvg), zeros);
            var absMdef = engine.TensorAbs(mdef);
            var positiveSigma = engine.TensorGreaterThan(sigmaMdef, NumOps.Zero);
            var lociScore = engine.TensorWhere(positiveSigma,
                engine.TensorDivide(absMdef, engine.TensorWhere(positiveSigma, sigmaMdef, ones)), absMdef);

            // Isolated (counting neighbours but no sampling neighbours) scores IsolatedPointScore; radii
            // with fewer than 2 counting neighbours are skipped
            var candidate = engine.TensorWhere(hasSampling, lociScore, engine.TensorAddScalar(zeros, isolatedScore));
            maxMdef = engine.TensorWhere(
                engine.TensorMultiply(counted, engine.TensorGreaterThan(candidate, maxMdef)), candidate, maxMdef);
        }

        // A point with no neighbours at ANY tested radius is an extreme isolate
        return engine.TensorWhere(hadNeighbors, maxMdef,
            engine.TensorAddScalar(zeros, NumOps.FromDouble(NoNeighborScore)));
    }

    // 1 where distance <= radius (a [1] tensor), else 0; not-greater-than, so NaN distances count as within.
    private Tensor<T> WithinRadius(Tensor<T> distances, Tensor<T> radius)
    {
        var engine = Engine;
        var radiusB = engine.TensorBroadcastTo(engine.Reshape(radius, new[] { 1, 1 }), new[] { distances.Shape[0], distances.Shape[1] });
        return engine.ScalarMinusTensor(NumOps.One, engine.TensorGreaterThan(distances, radiusB));
    }

    private T EstimateMaxRadius(Matrix<T> X)
    {
        // Estimate maximum radius as the max distance between sampled pairs of points
        // (pairs drawn in the seeded order idx1, idx2, idx1, idx2, ...)
        int sampleSize = Math.Min(100, X.Rows);
        var random = RandomHelper.CreateSeededRandom(_randomSeed);
        var first = new int[sampleSize];
        var second = new int[sampleSize];
        for (int i = 0; i < sampleSize; i++)
        {
            first[i] = random.Next(X.Rows);
            second[i] = random.Next(X.Rows);
        }

        var data = AnomalyTensorOps.ToTensor(X);
        var engine = Engine;
        var diff = engine.TensorSubtract(AnomalyTensorOps.SelectRows(data, first), AnomalyTensorOps.SelectRows(data, second));
        var distances = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(diff), new[] { 1 }, false));
        return engine.TensorMaxValue(engine.TensorMax(distances, NumOps.Zero));
    }
}
