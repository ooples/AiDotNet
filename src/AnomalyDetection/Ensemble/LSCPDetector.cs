using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Ensemble;

/// <summary>
/// Detects anomalies using LSCP (Locally Selective Combination in Parallel Outlier Ensembles).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> LSCP improves ensemble detection by selecting the most competent
/// detectors for each test point locally. Instead of combining all detectors equally,
/// it selects the best performing detectors based on local pseudo ground truth.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Train multiple diverse base detectors
/// 2. For each test point, find local region
/// 3. Evaluate detector competence in that region
/// 4. Combine scores from most competent detectors
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When different detectors work better in different regions
/// - For heterogeneous anomaly types
/// - When ensemble averaging isn't optimal
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - N estimators: 10
/// - Local region size: 30 neighbors
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Zhao, Y., et al. (2019). "LSCP: Locally Selective Combination in Parallel
/// Outlier Ensembles." SDM.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("LSCP: Locally Selective Combination in Parallel Outlier Ensembles", "https://doi.org/10.1137/1.9781611975673.17", Year = 2019, Authors = "Yue Zhao, Zain Nasrullah, Maciej K. Hryniewicki, Zheng Li")]
public partial class LSCPDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _nEstimators;
    private readonly int _localRegionSize;
    private List<IAnomalyDetector<T>>? _baseDetectors;
    [Buffer]
    private Matrix<T>? _trainingData;
    [Buffer]
    private Vector<T>[]? _detectorScores;

    /// <summary>
    /// Gets the number of estimators.
    /// </summary>
    public int NEstimators => _nEstimators;

    /// <summary>
    /// Gets the local region size.
    /// </summary>
    public int LocalRegionSize => _localRegionSize;

    /// <summary>
    /// Creates a new LSCP anomaly detector.
    /// </summary>
    /// <param name="nEstimators">Number of base detectors. Default is 10.</param>
    /// <param name="localRegionSize">
    /// Number of neighbors for local region. Default is 30.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public LSCPDetector(int nEstimators = 10, int localRegionSize = 30,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (nEstimators < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(nEstimators),
                "NEstimators must be at least 2. Recommended is 10.");
        }

        if (localRegionSize < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(localRegionSize),
                "LocalRegionSize must be at least 2. Recommended is 30.");
        }

        _nEstimators = nEstimators;
        _localRegionSize = localRegionSize;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;

        _trainingData = X;

        // Create diverse base detectors
        _baseDetectors = new List<IAnomalyDetector<T>>();
        _detectorScores = new Vector<T>[_nEstimators];

        int k = Math.Min(10, n - 1);

        for (int e = 0; e < _nEstimators; e++)
        {
            IAnomalyDetector<T> detector;

            // Create diverse detector types
            switch (e % 3)
            {
                case 0:
                    detector = new DistanceBased.LocalOutlierFactor<T>(
                        numNeighbors: Math.Max(1, k + e - 1),
                        contamination: _contamination,
                        randomSeed: _randomSeed + e);
                    break;
                case 1:
                    detector = new DistanceBased.KNNDetector<T>(
                        k: Math.Max(1, k + e - 1),
                        contamination: _contamination,
                        randomSeed: _randomSeed + e);
                    break;
                default:
                    detector = new TreeBased.IsolationForest<T>(
                        numTrees: 50 + e * 10,
                        maxSamples: Math.Min(256, n),
                        contamination: _contamination,
                        randomSeed: _randomSeed + e);
                    break;
            }

            detector.Fit(X);
            _baseDetectors.Add(detector);

            // Store training scores
            _detectorScores[e] = detector.ScoreAnomalies(X);
        }

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

        var trainingData = _trainingData;
        var baseDetectors = _baseDetectors;
        if (trainingData == null || baseDetectors == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var detScores = _detectorScores ?? throw new InvalidOperationException("Model not properly fitted.");
        if (X.Columns != trainingData.Columns)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {trainingData.Columns} features.",
                nameof(X));
        }

        // Zhao et al. (2019) / PyOD LSCP. Detector scores on the query rows and on the training rows,
        // standardized with the training mean and (population) std of each detector (a zero std counts as 1)
        var engine = Engine;
        int m = X.Rows;
        int n = trainingData.Rows;
        var testScores = new Tensor<T>[_nEstimators];
        for (int e = 0; e < _nEstimators; e++)
        {
            testScores[e] = baseDetectors[e].ScoreAnomalies(X).AsTensor();
        }

        var trainRaw = engine.TensorStack(detScores.Select(s => s.AsTensor()).ToArray(), 1);
        var testRaw = engine.TensorStack(testScores, 1);
        var (std, mean) = engine.TensorStdMean(trainRaw, new[] { 0 }, 0, false);
        var scale = engine.TensorWhere(engine.TensorEquals(std, NumOps.Zero), Tensor<T>.CreateOnes(new[] { _nEstimators }), std);
        var trainNorm = AnomalyTensorOps.DivideRow(AnomalyTensorOps.SubtractRow(trainRaw, mean), scale);
        var testNorm = AnomalyTensorOps.DivideRow(AnomalyTensorOps.SubtractRow(testRaw, mean), scale);

        // Pseudo ground truth: the largest standardized training score of each training row
        var pseudo = engine.ReduceMax(trainNorm, new[] { 1 }, false);

        // Query rows are combined in bounded blocks so that the [rows, n] region and correlation tensors
        // only ever cover one block; the feature subspaces are drawn once and shared by every block
        if (m == 0)
        {
            return new Vector<T>(0);
        }

        var featureSets = DrawFeatureSubspaces(trainingData.Columns);
        var query = AnomalyTensorOps.ToTensor(X);
        var reference = AnomalyTensorOps.ToTensor(trainingData);
        int rowsPerBlock = Math.Max(1, Math.Min(m, RegionBlockElements / Math.Max(1, n)));
        var blockScores = new List<Tensor<T>>();
        for (int first = 0; first < m; first += rowsPerBlock)
        {
            int rows = Math.Min(rowsPerBlock, m - first);
            var queryBlock = engine.TensorSlice(query, new[] { first, 0 }, new[] { rows, query.Shape[1] });
            var testBlock = engine.TensorSlice(testNorm, new[] { first, 0 }, new[] { rows, _nEstimators });
            blockScores.Add(CombineBlock(LocalRegionsBlock(queryBlock, reference, featureSets), trainNorm, pseudo, testBlock));
        }

        return (blockScores.Count == 1 ? blockScores[0] : engine.TensorConcatenate(blockScores.ToArray(), 0)).ToVector();
    }

    // Upper bound on the elements of one [rows, n] block of local-region tensors.
    private const int RegionBlockElements = 1 << 22;

    // The LSCP combination of one block of query rows: region is the block's 0/1 local-region mask
    // ([m, n]), testNorm its standardized detector scores ([m, e]).
    private Tensor<T> CombineBlock(Tensor<T> region, Tensor<T> trainNorm, Tensor<T> pseudo, Tensor<T> testNorm)
    {
        var engine = Engine;
        int m = region.Shape[0];

        // Competence of every detector: Pearson correlation of its standardized training scores with the
        // pseudo ground truth over the local region of the query row (0 where undefined)
        var count = engine.ReduceSum(region, new[] { 1 }, false);
        var pseudoRows = AnomalyTensorOps.BroadcastRow(pseudo, m);
        var pseudoCentered = engine.TensorMultiply(region,
            AnomalyTensorOps.SubtractColumn(pseudoRows, engine.TensorDivide(engine.ReduceSum(engine.TensorMultiply(region, pseudoRows), new[] { 1 }, false), count)));
        var pseudoSquares = engine.ReduceSum(engine.TensorSquare(pseudoCentered), new[] { 1 }, false);
        var correlations = new Tensor<T>[_nEstimators];
        for (int e = 0; e < _nEstimators; e++)
        {
            var detectorRows = AnomalyTensorOps.BroadcastRow(engine.TensorSliceAxis(trainNorm, 1, e), m);
            var detectorCentered = engine.TensorMultiply(region, AnomalyTensorOps.SubtractColumn(detectorRows,
                engine.TensorDivide(engine.ReduceSum(engine.TensorMultiply(region, detectorRows), new[] { 1 }, false), count)));
            var covariance = engine.ReduceSum(engine.TensorMultiply(pseudoCentered, detectorCentered), new[] { 1 }, false);
            var denominator = engine.TensorSqrt(engine.TensorMultiply(pseudoSquares,
                engine.ReduceSum(engine.TensorSquare(detectorCentered), new[] { 1 }, false)));
            var defined = engine.TensorGreaterThan(denominator, NumOps.Zero);
            correlations[e] = engine.TensorWhere(defined,
                engine.TensorDivide(covariance, engine.TensorWhere(defined, denominator, Tensor<T>.CreateOnes(new[] { m }))),
                Tensor<T>.CreateZeros(new[] { m }));
        }

        // Average of the standardized query scores of the competent detectors (the most populated
        // histogram bin of the correlations)
        var competent = CompetentDetectors(engine.TensorStack(correlations, 1));
        var selectedScores = engine.ReduceSum(engine.TensorMultiply(competent, testNorm), new[] { 1 }, false);
        return engine.TensorDivide(selectedScores, engine.ReduceSum(competent, new[] { 1 }, false));
    }

    // PyOD _get_local_region: over 20 random feature subspaces (between half and all but one of the
    // features, all features when there is only one), the localRegionSize nearest training rows of each
    // query row (ties to the lower index); the region is the rows found more than 10 times, the bar
    // lowered one step at a time until at least 2 rows qualify. 0/1 mask [m, n]. The subspaces come
    // from a generator seeded with the model seed on every call, so scores are reproducible.
    internal Tensor<T> LocalRegions(Matrix<T> X, Matrix<T> trainingData)
    {
        var featureSets = DrawFeatureSubspaces(trainingData.Columns);
        var reference = AnomalyTensorOps.ToTensor(trainingData);
        return AnomalyTensorOps.MapRowBlocks(AnomalyTensorOps.ToTensor(X), trainingData.Rows,
            queryBlock => LocalRegionsBlock(queryBlock, reference, featureSets));
    }

    private const int RegionIterations = 20;

    // The RegionIterations feature subspaces, from a generator seeded with the model seed on every call.
    private int[][] DrawFeatureSubspaces(int d)
    {
        var random = RandomHelper.CreateSeededRandom(_randomSeed);
        int minFeatures = (int)(d * 0.5);
        bool allFeatures = minFeatures < 1;
        var featureSets = new int[RegionIterations][];
        for (int iteration = 0; iteration < RegionIterations; iteration++)
        {
            int[] features;
            if (allFeatures)
            {
                features = Enumerable.Range(0, d).ToArray();
            }
            else
            {
                // numpy randint(min, d) (exclusive upper bound), then a draw without replacement
                int count = random.Next(minFeatures, d);
                var pool = Enumerable.Range(0, d).ToArray();
                for (int i = 0; i < count; i++)
                {
                    int j = i + random.Next(d - i);
                    (pool[i], pool[j]) = (pool[j], pool[i]);
                }

                features = new int[count];
                Array.Copy(pool, features, count);
            }

            featureSets[iteration] = features;
        }

        return featureSets;
    }

    // Local-region mask ([m, n]) of a block of query rows over the given feature subspaces.
    private Tensor<T> LocalRegionsBlock(Tensor<T> query, Tensor<T> reference, int[][] featureSets)
    {
        const int Threshold = RegionIterations / 2;
        var engine = Engine;
        int m = query.Shape[0];
        int n = reference.Shape[0];
        int k = Math.Min(_localRegionSize, n);
        var ones = Tensor<T>.CreateOnes(new[] { m, k });
        var counts = Tensor<T>.CreateZeros(new[] { m, n });

        foreach (var features in featureSets)
        {
            var distances = AnomalyTensorOps.PairwiseDistances(
                AnomalyTensorOps.SelectColumns(query, features), AnomalyTensorOps.SelectColumns(reference, features));
            var (_, order) = AnomalyTensorOps.StableSortRows(distances, descending: false);
            var nearest = engine.TensorSlice(order, new[] { 0, 0 }, new[] { m, k });
            counts = engine.TensorAdd(counts, engine.Scatter(Tensor<T>.CreateZeros(new[] { m, n }), nearest, ones, 1));
        }

        // Bar per row: Threshold, or (second largest count - 1) when fewer than 2 rows exceed Threshold
        var (sortedCounts, _) = engine.TensorSort(counts, 1, true);
        var second = engine.TensorSliceAxis(sortedCounts, 1, Math.Min(1, n - 1));
        var bar = engine.TensorMin(engine.TensorSubtractScalar(second, NumOps.One),
            engine.Fill(m, NumOps.FromDouble(Threshold)).AsTensor());
        return engine.TensorGreaterThan(counts, AnomalyTensorOps.BroadcastColumn(bar, n));
    }

    // PyOD _get_competent_detectors: a numpy histogram of each row of correlations ([m, e]) with
    // min(10, e) equal-width bins over [min, max] ([min - 0.5, max + 0.5] when they are equal); the
    // detectors whose correlation lies in the closed range of the most populated bin (the last such bin
    // on ties) are competent. 0/1 mask [m, e].
    internal Tensor<T> CompetentDetectors(Tensor<T> correlations)
    {
        var engine = Engine;
        int m = correlations.Shape[0];
        int e = correlations.Shape[1];
        int bins = Math.Min(10, e);
        T half = NumOps.FromDouble(0.5);
        var axis1 = new[] { 1 };

        var low = engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(correlations), axis1, false));
        var high = engine.ReduceMax(correlations, axis1, false);
        var flat = engine.TensorEquals(low, high);
        low = engine.TensorWhere(flat, engine.TensorSubtractScalar(low, half), low);
        high = engine.TensorWhere(flat, engine.TensorAddScalar(high, half), high);

        // Edges low + b * (high - low) / bins, the last one exactly high (numpy linspace)
        var step = engine.TensorDivideScalar(engine.TensorSubtract(high, low), NumOps.FromDouble(bins));
        var positions = AnomalyTensorOps.BroadcastRow(AnomalyTensorOps.Range(bins + 1).Cast<T>(), m);
        var edges = engine.TensorAdd(AnomalyTensorOps.BroadcastColumn(low, bins + 1),
            engine.TensorMultiply(positions, AnomalyTensorOps.BroadcastColumn(step, bins + 1)));
        var lastEdge = engine.TensorEquals(positions, NumOps.FromDouble(bins));
        edges = engine.TensorWhere(lastEdge, AnomalyTensorOps.BroadcastColumn(high, bins + 1), edges);

        // Bin of each value: the number of interior edges at or below it (numpy puts the maximum in the last bin)
        var shape = new[] { m, e, bins + 1 };
        var values3 = engine.TensorBroadcastTo(engine.Reshape(correlations, new[] { m, e, 1 }), shape);
        var edges3 = engine.TensorBroadcastTo(engine.Reshape(edges, new[] { m, 1, bins + 1 }), shape);
        var atOrAbove = engine.ScalarMinusTensor(NumOps.One, engine.TensorLessThan(values3, edges3));
        var interior = engine.TensorBroadcastTo(engine.Reshape(
            engine.TensorMultiply(engine.TensorGreaterThan(AnomalyTensorOps.Range(bins + 1).Cast<T>(), NumOps.Zero),
                engine.TensorLessThan(AnomalyTensorOps.Range(bins + 1).Cast<T>(), NumOps.FromDouble(bins))),
            new[] { 1, 1, bins + 1 }), shape);
        var binIndex = engine.ReduceSum(engine.TensorMultiply(atOrAbove, interior), new[] { 2 }, false);

        // Histogram counts and the most populated bin (last among equals)
        var binPositions = engine.TensorBroadcastTo(engine.Reshape(AnomalyTensorOps.Range(bins).Cast<T>(), new[] { 1, 1, bins }),
            new[] { m, e, bins });
        var oneHot = engine.TensorEquals(engine.TensorBroadcastTo(engine.Reshape(binIndex, new[] { m, e, 1 }), new[] { m, e, bins }),
            binPositions);
        var histogram = engine.ReduceSum(oneHot, axis1, false);
        var best = AnomalyTensorOps.LastArgMaxRows(histogram);

        // Competent: edge[best] <= correlation <= edge[best + 1]
        var bestColumn = engine.Reshape(best, new[] { m, 1 });
        var lower = engine.TensorTakeAlongDim(edges, bestColumn, 1);
        var upper = engine.TensorTakeAlongDim(edges, engine.TensorAddScalar(bestColumn, 1), 1);
        var belowLower = engine.TensorLessThan(correlations, engine.TensorBroadcastTo(lower, new[] { m, e }));
        var aboveUpper = engine.TensorGreaterThan(correlations, engine.TensorBroadcastTo(upper, new[] { m, e }));
        return engine.TensorMultiply(engine.ScalarMinusTensor(NumOps.One, belowLower),
            engine.ScalarMinusTensor(NumOps.One, aboveUpper));
    }
}
