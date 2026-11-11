using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Implements core-set selection using the k-center-greedy algorithm for active learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Core-set selection aims to select samples that best represent
/// the overall data distribution. It uses the k-center algorithm which iteratively selects
/// the sample that is farthest from all previously selected samples. This ensures good
/// coverage of the feature space.</para>
///
/// <para><b>Algorithm (k-center-greedy):</b></para>
/// <list type="number">
/// <item><description>Start with a random sample or the one farthest from the origin.</description></item>
/// <item><description>Repeat until batch is full:
///   <list type="bullet">
///   <item><description>For each remaining sample, compute distance to nearest selected sample.</description></item>
///   <item><description>Select the sample with maximum minimum-distance.</description></item>
///   </list>
/// </description></item>
/// </list>
///
/// <para><b>Advantages:</b></para>
/// <list type="bullet">
/// <item><description>Ensures diverse coverage of feature space.</description></item>
/// <item><description>Works independently of model predictions.</description></item>
/// <item><description>Good for initial exploration before model is trained.</description></item>
/// </list>
///
/// <para><b>Complexity:</b> O(n × k × d) where n=pool size, k=batch size, d=feature dimension.</para>
/// <para><b>Reference:</b> Sener &amp; Savarese, "Active Learning for Convolutional Neural Networks:
/// A Core-Set Approach" (ICLR 2018).</para>
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
public class CoreSetSelection<T> : IActiveLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private bool _useBatchDiversity;
    private T _lastMinScore;
    private T _lastMaxScore;
    private T _lastMeanScore;

    /// <summary>
    /// Defines the distance metric to use for core-set selection.
    /// </summary>
    public enum DistanceMetric
    {
        /// <summary>Standard Euclidean (L2) distance.</summary>
        Euclidean,
        /// <summary>Manhattan (L1) distance.</summary>
        Manhattan,
        /// <summary>Cosine distance (1 - cosine similarity).</summary>
        Cosine
    }

    private readonly DistanceMetric _distanceMetric;

    /// <summary>
    /// Initializes a new instance of the CoreSetSelection class.
    /// </summary>
    /// <param name="distanceMetric">The distance metric to use (default: Euclidean).</param>
    public CoreSetSelection(DistanceMetric distanceMetric = DistanceMetric.Euclidean)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _distanceMetric = distanceMetric;
        _useBatchDiversity = true; // Core-set is inherently diversity-based
        _lastMinScore = _numOps.Zero;
        _lastMaxScore = _numOps.Zero;
        _lastMeanScore = _numOps.Zero;
    }

    /// <inheritdoc />
    public string Name => $"CoreSetSelection-{_distanceMetric}";

    /// <inheritdoc />
    public bool UseBatchDiversity
    {
        get => _useBatchDiversity;
        set => _useBatchDiversity = value; // Note: Core-set always uses diversity inherently
    }

    /// <inheritdoc />
    public int[] SelectSamples(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool, int batchSize)
    {
        Guard.NotNull(model);
        Guard.NotNull(unlabeledPool);

        var numSamples = unlabeledPool.Shape[0];
        batchSize = Math.Min(batchSize, numSamples);
        var featureSize = unlabeledPool.Length / numSamples;

        // Use k-center-greedy algorithm
        return KCenterGreedy(unlabeledPool, featureSize, batchSize);
    }

    /// <inheritdoc />
    public Vector<T> ComputeInformativenessScores(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool)
    {
        Guard.NotNull(model);
        Guard.NotNull(unlabeledPool);

        var numSamples = unlabeledPool.Shape[0];
        var featureSize = unlabeledPool.Length / numSamples;
        var rows = ActiveLearningEngineOps.AsRows(unlabeledPool, numSamples, featureSize);

        // Score each sample by its distance to the center of the data
        var center = ComputeDataCenter(rows, numSamples);
        var scores = ActiveLearningEngineOps.DistancesToPoint(rows, center).ToVector();

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
            ["MeanScore"] = _lastMeanScore
        };
    }

    /// <summary>
    /// Implements the k-center-greedy algorithm.
    /// </summary>
    private int[] KCenterGreedy(Tensor<T> pool, int featureSize, int batchSize)
    {
        var engine = AiDotNetEngine.Current;
        var numSamples = pool.Shape[0];
        var rows = ActiveLearningEngineOps.AsRows(pool, numSamples, featureSize);
        var norms = _distanceMetric == DistanceMetric.Cosine ? ActiveLearningEngineOps.RowNorms(rows) : null;
        var selected = new List<int>();
        var excluded = new bool[numSamples];

        // Start with the sample farthest from origin (or we could use random)
        var firstIdx = FindFarthestFromOrigin(rows);
        selected.Add(firstIdx);
        excluded[firstIdx] = true;

        // Track minimum distance to selected set for each sample
        var minDistToSelected = ComputeDistances(rows, norms, firstIdx).Reshape(new[] { 1, numSamples });

        // Iteratively select sample with maximum min-distance to selected set
        while (selected.Count < batchSize && selected.Count < numSamples)
        {
            var best = ActiveLearningEngineOps.ArgMaxAbove(minDistToSelected, excluded, _numOps.MinValue);
            if (best < 0)
            {
                // No remaining sample has a comparable distance (e.g. NaN): stop instead of spinning.
                break;
            }

            selected.Add(best);
            excluded[best] = true;

            // Update min distances
            var dist = ComputeDistances(rows, norms, best).Reshape(new[] { 1, numSamples });
            minDistToSelected = engine.TensorMin(minDistToSelected, dist);
        }

        return [.. selected];
    }

    /// <summary>
    /// Finds the sample farthest from the origin.
    /// </summary>
    private int FindFarthestFromOrigin(Tensor<T> rows)
    {
        var engine = AiDotNetEngine.Current;
        var numSamples = rows.Shape[0];
        var squaredNorms = rows.Shape[1] == 0
            ? new Tensor<T>(new[] { numSamples })
            : engine.ReduceSum(engine.TensorMultiply(rows, rows), new[] { 1 }, keepDims: false);
        var maxIdx = ActiveLearningEngineOps.ArgMaxAbove(squaredNorms.Reshape(new[] { 1, numSamples }), new bool[numSamples], _numOps.MinValue);
        return maxIdx < 0 ? 0 : maxIdx;
    }

    /// <summary>
    /// Computes the distance from sample <paramref name="index"/> to every sample based on the
    /// configured metric.
    /// </summary>
    /// <returns>A [numSamples] tensor.</returns>
    private Tensor<T> ComputeDistances(Tensor<T> rows, Tensor<T>? norms, int index)
    {
        return _distanceMetric switch
        {
            DistanceMetric.Manhattan => ComputeManhattanDistances(rows, index),
            DistanceMetric.Cosine => ComputeCosineDistances(rows, norms ?? ActiveLearningEngineOps.RowNorms(rows), index),
            _ => ActiveLearningEngineOps.DistancesToRow(rows, index)
        };
    }

    /// <summary>
    /// Computes Manhattan distances from one sample to every sample.
    /// </summary>
    private static Tensor<T> ComputeManhattanDistances(Tensor<T> rows, int index)
    {
        var engine = AiDotNetEngine.Current;
        int numSamples = rows.Shape[0];
        int featureSize = rows.Shape[1];
        if (featureSize == 0)
        {
            return new Tensor<T>(new[] { numSamples });
        }

        var point = engine.TensorSlice(rows, new[] { index, 0 }, new[] { 1, featureSize });
        var diff = engine.TensorSubtract(rows, engine.TensorBroadcastTo(point, new[] { numSamples, featureSize }));
        return engine.ReduceSum(engine.TensorAbs(diff), new[] { 1 }, keepDims: false);
    }

    /// <summary>
    /// Computes cosine distances (1 - cosine_similarity) from one sample to every sample.
    /// </summary>
    private Tensor<T> ComputeCosineDistances(Tensor<T> rows, Tensor<T> norms, int index)
    {
        var engine = AiDotNetEngine.Current;
        int numSamples = rows.Shape[0];
        var dotProduct = ActiveLearningEngineOps.DotBlock(rows, index, 1).Reshape(new[] { numSamples });
        var indexNorm = engine.TensorSlice(norms, new[] { index }, new[] { 1 });
        var normProduct = engine.TensorMultiply(norms, engine.TensorBroadcastTo(indexNorm, new[] { numSamples }));

        var cosineDistance = engine.ScalarMinusTensor(_numOps.One, engine.TensorDivide(dotProduct, normProduct));
        var ones = new Tensor<T>(new[] { numSamples });
        ones.Fill(_numOps.One);
        var degenerate = engine.TensorLessThan(normProduct, _numOps.FromDouble(1e-10));
        return engine.TensorWhere(degenerate, ones, cosineDistance);
    }

    /// <summary>
    /// Computes the center of the data.
    /// </summary>
    /// <returns>A [1, featureSize] tensor.</returns>
    private Tensor<T> ComputeDataCenter(Tensor<T> rows, int numSamples)
    {
        var engine = AiDotNetEngine.Current;
        var sum = engine.ReduceSum(rows, new[] { 0 }, keepDims: true);
        return engine.TensorDivideScalar(sum, _numOps.FromDouble(numSamples));
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
