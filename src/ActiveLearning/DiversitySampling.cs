using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Implements Diversity Sampling for active learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Diversity Sampling selects samples that are representative of
/// different regions in the input space. Instead of focusing on uncertain samples near the
/// decision boundary, diversity sampling ensures good coverage of the data distribution.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>Compute pairwise distances between all unlabeled samples.</description></item>
/// <item><description>Select samples that maximize diversity - samples that are far apart
/// from each other and from already labeled data.</description></item>
/// <item><description>Can use k-medoids clustering, farthest-first traversal, or core-set
/// construction approaches.</description></item>
/// </list>
///
/// <para><b>Strategies:</b></para>
/// <list type="bullet">
/// <item><description><b>Farthest-First:</b> Greedily selects the sample farthest from all
/// previously selected samples.</description></item>
/// <item><description><b>K-Center-Greedy:</b> Selects samples to minimize the maximum distance
/// from any point to its nearest selected sample (core-set construction).</description></item>
/// <item><description><b>Clustering-Based:</b> Clusters the data and selects samples closest
/// to cluster centers.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Sener and Savarese, "Active Learning for Convolutional Neural Networks:
/// A Core-Set Approach" (2018). ICLR.</para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Optimization)]
[ModelCategory(ModelCategory.Clustering)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Active Learning for Convolutional Neural Networks: A Core-Set Approach", "https://arxiv.org/abs/1708.00489", Year = 2018, Authors = "Ozan Sener, Silvio Savarese")]
[ComponentType(ComponentType.ActiveLearner)]
[PipelineStage(PipelineStage.Training)]
public class DiversitySampling<T> : IActiveLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly DiversityMethod _method;
    private readonly DistanceMetric _distanceMetric;
    private bool _useBatchDiversity;
    private T _lastMinScore;
    private T _lastMaxScore;
    private T _lastMeanScore;
    private T _lastCoverageRadius;

    /// <summary>
    /// Defines the diversity selection method.
    /// </summary>
    public enum DiversityMethod
    {
        /// <summary>Greedily selects farthest sample from current selection.</summary>
        FarthestFirst,
        /// <summary>K-center greedy for core-set construction.</summary>
        KCenterGreedy,
        /// <summary>Select samples based on density peaks.</summary>
        DensityPeaks
    }

    /// <summary>
    /// Defines the distance metric to use.
    /// </summary>
    public enum DistanceMetric
    {
        /// <summary>Euclidean distance.</summary>
        Euclidean,
        /// <summary>Cosine distance (1 - cosine similarity).</summary>
        Cosine,
        /// <summary>Manhattan (L1) distance.</summary>
        Manhattan
    }

    /// <summary>
    /// Initializes a new instance of the DiversitySampling class.
    /// </summary>
    /// <param name="method">The diversity method to use (default: KCenterGreedy).</param>
    /// <param name="distanceMetric">The distance metric to use (default: Euclidean).</param>
    public DiversitySampling(
        DiversityMethod method = DiversityMethod.KCenterGreedy,
        DistanceMetric distanceMetric = DistanceMetric.Euclidean)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _method = method;
        _distanceMetric = distanceMetric;
        _useBatchDiversity = true; // Diversity sampling inherently considers batch diversity
        _lastMinScore = _numOps.Zero;
        _lastMaxScore = _numOps.Zero;
        _lastMeanScore = _numOps.Zero;
        _lastCoverageRadius = _numOps.Zero;
    }

    /// <inheritdoc />
    public string Name => $"DiversitySampling-{_method}-{_distanceMetric}";

    /// <inheritdoc />
    public bool UseBatchDiversity
    {
        get => _useBatchDiversity;
        set => _useBatchDiversity = value;
    }

    /// <summary>
    /// Gets the coverage radius from the last selection.
    /// </summary>
    /// <remarks>
    /// <para>The coverage radius is the maximum distance from any point to its nearest
    /// selected sample. Lower values indicate better coverage of the data space.</para>
    /// </remarks>
    public T CoverageRadius => _lastCoverageRadius;

    /// <inheritdoc />
    public int[] SelectSamples(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool, int batchSize)
    {
        // Note: model is not used in pure diversity sampling, but we accept it for interface compatibility
        Guard.NotNull(unlabeledPool);

        var numSamples = unlabeledPool.Shape[0];
        batchSize = Math.Min(batchSize, numSamples);

        var selected = _method switch
        {
            DiversityMethod.FarthestFirst => SelectFarthestFirst(unlabeledPool, batchSize),
            DiversityMethod.KCenterGreedy => SelectKCenterGreedy(unlabeledPool, batchSize),
            DiversityMethod.DensityPeaks => SelectDensityPeaks(unlabeledPool, batchSize),
            _ => SelectKCenterGreedy(unlabeledPool, batchSize)
        };

        // Compute coverage radius
        _lastCoverageRadius = ComputeCoverageRadius(unlabeledPool, selected);

        return selected;
    }

    /// <inheritdoc />
    public Vector<T> ComputeInformativenessScores(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool)
    {
        // Note: model is not used in diversity sampling
        Guard.NotNull(unlabeledPool);

        var numSamples = unlabeledPool.Shape[0];
        var featureSize = unlabeledPool.Length / numSamples;

        // Compute diversity scores based on distance to nearest neighbors
        var scores = new Vector<T>(numSamples);

        if (_method == DiversityMethod.DensityPeaks)
        {
            // Density-based scoring: high density samples are more representative
            scores = ComputeDensityScores(unlabeledPool, numSamples, featureSize);
        }
        else
        {
            // Distance-based scoring: samples far from others get high scores
            scores = ComputeDistanceScores(unlabeledPool, numSamples, featureSize);
        }

        UpdateStatistics(scores);
        return scores;
    }

    /// <inheritdoc />
    public Dictionary<string, T> GetSelectionStatistics()
    {
        return new Dictionary<string, T>
        {
            ["MinScore"] = _lastMinScore,
            ["MaxScore"] = _lastMaxScore,
            ["MeanScore"] = _lastMeanScore,
            ["CoverageRadius"] = _lastCoverageRadius
        };
    }

    /// <summary>
    /// Selects samples using farthest-first traversal.
    /// </summary>
    private int[] SelectFarthestFirst(Tensor<T> pool, int batchSize)
    {
        var engine = AiDotNetEngine.Current;
        var numSamples = pool.Shape[0];
        var featureSize = pool.Length / numSamples;
        var rows = ActiveLearningEngineOps.AsRows(pool, numSamples, featureSize);
        var norms = CosineNorms(rows);
        var selected = new List<int>();
        var excluded = new bool[numSamples];

        // Start with a random sample (or the centroid's nearest neighbor)
        var centroid = ComputeCentroid(rows, numSamples);
        var first = FindNearestToCentroid(rows, centroid);
        selected.Add(first);
        excluded[first] = true;

        // Minimum distance from every sample to the selected set
        var minDist = new Tensor<T>(new[] { 1, numSamples });
        minDist.Fill(_numOps.MaxValue);
        minDist = engine.TensorMin(minDist, ComputeDistanceBlock(rows, norms, first, 1));

        // Greedily add the farthest sample
        while (selected.Count < batchSize && selected.Count < numSamples)
        {
            var farthest = ActiveLearningEngineOps.ArgMaxAbove(minDist, excluded, _numOps.MinValue);
            if (farthest < 0)
            {
                // No remaining sample has a comparable distance (e.g. NaN): stop instead of spinning.
                break;
            }

            selected.Add(farthest);
            excluded[farthest] = true;
            minDist = engine.TensorMin(minDist, ComputeDistanceBlock(rows, norms, farthest, 1));
        }

        return [.. selected];
    }

    /// <summary>
    /// Selects samples using k-center greedy algorithm for core-set construction.
    /// </summary>
    private int[] SelectKCenterGreedy(Tensor<T> pool, int batchSize)
    {
        // K-center greedy is similar to farthest-first but specifically aims to
        // minimize the maximum distance from any point to the nearest center
        return SelectFarthestFirst(pool, batchSize);
    }

    /// <summary>
    /// Selects samples based on density peaks.
    /// </summary>
    private int[] SelectDensityPeaks(Tensor<T> pool, int batchSize)
    {
        var engine = AiDotNetEngine.Current;
        var numSamples = pool.Shape[0];
        var featureSize = pool.Length / numSamples;
        var rows = ActiveLearningEngineOps.AsRows(pool, numSamples, featureSize);
        var norms = CosineNorms(rows);

        // Compute local density for each sample
        var densities = ComputeLocalDensities(rows, norms);

        // Compute distance to nearest higher-density sample
        var deltaDistances = ComputeDeltaDistances(rows, norms, densities);

        // Score = density * delta (decision graph)
        var scores = engine.TensorMultiply(densities, deltaDistances).ToVector();

        // Select top-scoring samples (density peaks)
        return ActiveLearningEngineOps.SelectTopScoring(scores, batchSize);
    }

    /// <summary>
    /// Computes local density for each sample using a Gaussian kernel.
    /// </summary>
    private Tensor<T> ComputeLocalDensities(Tensor<T> rows, Tensor<T>? norms)
    {
        var engine = AiDotNetEngine.Current;
        int numSamples = rows.Shape[0];

        // Compute average distance to estimate bandwidth
        var avgDist = ComputeAverageNearestNeighborDistance(rows, norms, k: 5);
        var minBandwidth = ActiveLearningEngineOps.Tolerance<T>(1e-10);
        var bandwidth = _numOps.GreaterThan(avgDist, minBandwidth) ? avgDist : minBandwidth;

        var minusHalf = _numOps.FromDouble(-0.5);
        var blockRows = ActiveLearningEngineOps.PairwiseBlockRows(numSamples, rows.Shape[1]);
        var densities = new List<Tensor<T>>();
        for (int start = 0; start < numSamples; start += blockRows)
        {
            int count = Math.Min(blockRows, numSamples - start);
            var dist = ComputeDistanceBlock(rows, norms, start, count);
            // exp(-(d / h)^2 / 2): dividing before squaring keeps d^2 and 2 h^2 from overflowing in Half
            var scaled = engine.TensorDivideScalar(dist, bandwidth);
            var kernel = engine.TensorExp(engine.TensorMultiplyScalar(engine.TensorMultiply(scaled, scaled), minusHalf));
            var others = engine.TensorMaskedFill(kernel, ActiveLearningEngineOps.SelfMask(start, count, numSamples), _numOps.Zero);
            densities.Add(engine.ReduceSum(others, new[] { 1 }, keepDims: false));
        }

        return engine.TensorConcatenate(densities.ToArray(), 0);
    }

    /// <summary>
    /// Computes delta distance (distance to nearest higher-density sample).
    /// </summary>
    private Tensor<T> ComputeDeltaDistances(Tensor<T> rows, Tensor<T>? norms, Tensor<T> densities)
    {
        var engine = AiDotNetEngine.Current;
        int numSamples = rows.Shape[0];
        var blockRows = ActiveLearningEngineOps.PairwiseBlockRows(numSamples, rows.Shape[1]);
        var densityRow = densities.Reshape(new[] { 1, numSamples });
        var deltas = new List<Tensor<T>>();

        for (int start = 0; start < numSamples; start += blockRows)
        {
            int count = Math.Min(blockRows, numSamples - start);
            var shape = new[] { count, numSamples };
            var dist = ComputeDistanceBlock(rows, norms, start, count);

            // Distance to the nearest sample with strictly higher density (a sample is never
            // denser than itself)
            var ownDensity = engine.TensorSlice(densities, new[] { start }, new[] { count }).Reshape(new[] { count, 1 });
            var higher = engine.TensorGreaterThan(engine.TensorBroadcastTo(densityRow, shape), engine.TensorBroadcastTo(ownDensity, shape));
            var unreachable = new Tensor<T>(shape);
            unreachable.Fill(_numOps.MaxValue);
            var minDistToHigherDensity = engine.TensorAmin(engine.TensorWhere(higher, dist, unreachable), new[] { 1 }, keepDims: false);
            var foundHigher = engine.TensorLessThan(minDistToHigherDensity, _numOps.MaxValue);

            // If no higher density point, set delta to max distance
            var others = engine.TensorMaskedFill(dist, ActiveLearningEngineOps.SelfMask(start, count, numSamples), _numOps.Zero);
            var maxDist = engine.TensorMax(engine.ReduceMax(others, new[] { 1 }, keepDims: false), _numOps.Zero);

            deltas.Add(engine.TensorWhere(foundHigher, minDistToHigherDensity, maxDist));
        }

        return engine.TensorConcatenate(deltas.ToArray(), 0);
    }

    /// <summary>
    /// Computes distance-based scores (average distance to k nearest neighbors).
    /// </summary>
    private Vector<T> ComputeDistanceScores(Tensor<T> pool, int numSamples, int featureSize)
    {
        var engine = AiDotNetEngine.Current;
        var k = Math.Min(5, numSamples - 1);
        if (k <= 0)
        {
            // No neighbours: the empty average divides zero by zero, as the scalar formula did
            return engine.TensorDivideScalar(new Tensor<T>(new[] { numSamples }), _numOps.Zero).ToVector();
        }

        // Sort and take average of k nearest
        var rows = ActiveLearningEngineOps.AsRows(pool, numSamples, featureSize);
        var sums = SumOfNearestDistances(rows, CosineNorms(rows), k);

        // For outliers we want HIGH distance, so no inversion needed
        return engine.TensorDivideScalar(sums, _numOps.FromDouble(k)).ToVector();
    }

    /// <summary>
    /// Computes density-based scores (inverse of average distance = high density).
    /// </summary>
    private Vector<T> ComputeDensityScores(Tensor<T> pool, int numSamples, int featureSize)
    {
        var engine = AiDotNetEngine.Current;
        var k = Math.Min(5, numSamples - 1);
        Tensor<T> avgDist;
        if (k <= 0)
        {
            avgDist = engine.TensorDivideScalar(new Tensor<T>(new[] { numSamples }), _numOps.Zero);
        }
        else
        {
            var rows = ActiveLearningEngineOps.AsRows(pool, numSamples, featureSize);
            var sums = SumOfNearestDistances(rows, CosineNorms(rows), k);
            avgDist = engine.TensorDivideScalar(sums, _numOps.FromDouble(k));
        }

        // Density is inverse of distance
        var epsilon = ActiveLearningEngineOps.Tolerance<T>(1e-10);
        return engine.TensorReciprocal(engine.TensorAddScalar(avgDist, epsilon)).ToVector();
    }

    /// <summary>
    /// Computes the average k-nearest neighbor distance.
    /// </summary>
    private T ComputeAverageNearestNeighborDistance(Tensor<T> rows, Tensor<T>? norms, int k)
    {
        var engine = AiDotNetEngine.Current;
        int numSamples = rows.Shape[0];
        var taken = Math.Min(k, numSamples - 1);
        if (taken <= 0)
        {
            return _numOps.One;
        }

        var totalDist = engine.TensorSum(SumOfNearestDistances(rows, norms, taken));
        return _numOps.Divide(totalDist, _numOps.FromDouble((double)numSamples * taken));
    }

    /// <summary>
    /// Sum of the <paramref name="k"/> smallest distances from each sample to the other samples
    /// (1 &lt;= k &lt; numSamples).
    /// </summary>
    /// <returns>A [numSamples] tensor.</returns>
    private Tensor<T> SumOfNearestDistances(Tensor<T> rows, Tensor<T>? norms, int k)
    {
        var engine = AiDotNetEngine.Current;
        int numSamples = rows.Shape[0];
        var blockRows = ActiveLearningEngineOps.PairwiseBlockRows(numSamples, rows.Shape[1]);
        var sums = new List<Tensor<T>>();

        for (int start = 0; start < numSamples; start += blockRows)
        {
            int count = Math.Min(blockRows, numSamples - start);
            var dist = ComputeDistanceBlock(rows, norms, start, count);
            var others = engine.TensorMaskedFill(dist, ActiveLearningEngineOps.SelfMask(start, count, numSamples), _numOps.MaxValue);
            var (nearest, _) = engine.TopK(others, k, 1, false);
            sums.Add(engine.ReduceSum(nearest, new[] { 1 }, keepDims: false));
        }

        return engine.TensorConcatenate(sums.ToArray(), 0);
    }

    /// <summary>
    /// Computes the coverage radius for selected samples.
    /// </summary>
    private T ComputeCoverageRadius(Tensor<T> pool, int[] selected)
    {
        var engine = AiDotNetEngine.Current;
        var numSamples = pool.Shape[0];
        var featureSize = pool.Length / numSamples;
        var rows = ActiveLearningEngineOps.AsRows(pool, numSamples, featureSize);
        var norms = CosineNorms(rows);

        var minDist = new Tensor<T>(new[] { 1, numSamples });
        minDist.Fill(_numOps.MaxValue);
        foreach (var selIdx in selected)
        {
            minDist = engine.TensorMin(minDist, ComputeDistanceBlock(rows, norms, selIdx, 1));
        }

        var maxMinDist = engine.TensorMaxValue(minDist);
        return _numOps.GreaterThan(maxMinDist, _numOps.Zero) ? maxMinDist : _numOps.Zero;
    }

    /// <summary>
    /// Row norms needed by the cosine metric, or null for the other metrics.
    /// </summary>
    private Tensor<T>? CosineNorms(Tensor<T> rows)
    {
        return _distanceMetric == DistanceMetric.Cosine ? ActiveLearningEngineOps.RowNorms(rows) : null;
    }

    /// <summary>
    /// Computes distances between samples [start, start + count) and every sample based on the
    /// configured metric.
    /// </summary>
    /// <returns>A [count, numSamples] tensor.</returns>
    private Tensor<T> ComputeDistanceBlock(Tensor<T> rows, Tensor<T>? norms, int start, int count)
    {
        return _distanceMetric switch
        {
            DistanceMetric.Cosine => ComputeCosineDistance(rows, norms ?? ActiveLearningEngineOps.RowNorms(rows), start, count),
            DistanceMetric.Manhattan => ComputeManhattanDistance(rows, start, count),
            _ => ActiveLearningEngineOps.DistanceBlock(rows, start, count)
        };
    }

    /// <summary>
    /// Computes cosine distance (1 - cosine similarity); pairs where either norm is (near) zero
    /// get distance 1.
    /// </summary>
    private Tensor<T> ComputeCosineDistance(Tensor<T> rows, Tensor<T> norms, int start, int count)
    {
        var engine = AiDotNetEngine.Current;
        int numSamples = rows.Shape[0];
        var shape = new[] { count, numSamples };
        var norm1 = engine.TensorBroadcastTo(engine.TensorSlice(norms, new[] { start }, new[] { count }).Reshape(new[] { count, 1 }), shape);
        var norm2 = engine.TensorBroadcastTo(norms.Reshape(new[] { 1, numSamples }), shape);
        // Dot products of unit rows: a.b and |a| |b| overflow in Half
        var cosineSim = ActiveLearningEngineOps.CosineBlock(rows, norms, start, count);
        var distance = engine.ScalarMinusTensor(_numOps.One, cosineSim);

        var epsilon = ActiveLearningEngineOps.Tolerance<T>(1e-10);
        var degenerate = engine.TensorMax(engine.TensorLessThan(norm1, epsilon), engine.TensorLessThan(norm2, epsilon));
        var ones = new Tensor<T>(shape);
        ones.Fill(_numOps.One);
        return engine.TensorWhere(degenerate, ones, distance);
    }

    /// <summary>
    /// Computes Manhattan (L1) distance.
    /// </summary>
    private static Tensor<T> ComputeManhattanDistance(Tensor<T> rows, int start, int count)
    {
        var engine = AiDotNetEngine.Current;
        int numSamples = rows.Shape[0];
        int featureSize = rows.Shape[1];
        if (featureSize == 0)
        {
            return new Tensor<T>(new[] { count, numSamples });
        }

        var shape = new[] { count, numSamples, featureSize };
        var block = engine.TensorSlice(rows, new[] { start, 0 }, new[] { count, featureSize }).Reshape(new[] { count, 1, featureSize });
        var all = rows.Reshape(new[] { 1, numSamples, featureSize });
        var diff = engine.TensorSubtract(engine.TensorBroadcastTo(block, shape), engine.TensorBroadcastTo(all, shape));
        return engine.ReduceSum(engine.TensorAbs(diff), new[] { 2 }, keepDims: false);
    }

    /// <summary>
    /// Computes the centroid of the data.
    /// </summary>
    /// <returns>A [1, featureSize] tensor.</returns>
    private Tensor<T> ComputeCentroid(Tensor<T> rows, int numSamples)
    {
        return ActiveLearningEngineOps.MeanRow(rows);
    }

    /// <summary>
    /// Finds the sample nearest to the centroid (first one on ties).
    /// </summary>
    private int FindNearestToCentroid(Tensor<T> rows, Tensor<T> centroid)
    {
        var engine = AiDotNetEngine.Current;
        int numSamples = rows.Shape[0];

        // Distances rather than squared distances: same order, and they do not overflow in Half.
        // argmin over samples with dist < MaxValue == argmax of -dist above -MaxValue
        var dist = ActiveLearningEngineOps.DistancesToPoint(rows, centroid);
        var negated = engine.TensorNegate(dist).Reshape(new[] { 1, numSamples });
        var nearest = ActiveLearningEngineOps.ArgMaxAbove(negated, new bool[numSamples], _numOps.Negate(_numOps.MaxValue));
        return nearest < 0 ? 0 : nearest;
    }

    /// <summary>
    /// Updates selection statistics.
    /// </summary>
    private void UpdateStatistics(Vector<T> scores)
    {
        if (scores.Length == 0) return;

        (_lastMinScore, _lastMaxScore, _lastMeanScore) = ActiveLearningEngineOps.Statistics(scores);
    }
}
