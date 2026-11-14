using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Implements density-weighted sampling for active learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Density-weighted sampling combines uncertainty with density
/// weighting to avoid selecting outliers. Even if a sample is uncertain, it may not be
/// informative if it's an outlier that doesn't represent the data distribution.</para>
///
/// <para><b>Formula:</b> Score(x) = Uncertainty(x) × Density(x)^β</para>
/// <para>where Density(x) is computed using average distance to k nearest neighbors.</para>
///
/// <para><b>Parameters:</b></para>
/// <list type="bullet">
/// <item><description><b>β (beta):</b> Controls density influence. β=1 gives equal weight,
/// β>1 emphasizes density, β&lt;1 emphasizes uncertainty.</description></item>
/// <item><description><b>k:</b> Number of neighbors for density estimation.</description></item>
/// </list>
///
/// <para><b>Advantages:</b></para>
/// <list type="bullet">
/// <item><description>Avoids selecting outliers with high uncertainty.</description></item>
/// <item><description>Selects samples representative of the data distribution.</description></item>
/// <item><description>Configurable balance between uncertainty and density.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Settles, B. &amp; Craven, M. (2008). "An Analysis of Active Learning
/// Strategies for Sequence Labeling Tasks."</para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Optimization)]
[ModelCategory(ModelCategory.InstanceBased)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("An Analysis of Active Learning Strategies for Sequence Labeling Tasks", "https://aclanthology.org/D08-1112/", Year = 2008, Authors = "Burr Settles, Mark Craven")]
[ComponentType(ComponentType.ActiveLearner)]
[PipelineStage(PipelineStage.Training)]
public class DensityWeightedSampling<T> : IActiveLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly double _beta;
    private readonly int _kNeighbors;
    private bool _useBatchDiversity;
    private T _lastMinScore;
    private T _lastMaxScore;
    private T _lastMeanScore;

    /// <summary>
    /// Initializes a new instance of the DensityWeightedSampling class.
    /// </summary>
    /// <param name="beta">Exponent for density weighting (default: 1.0).</param>
    /// <param name="kNeighbors">Number of neighbors for density estimation (default: 10).</param>
    public DensityWeightedSampling(double beta = 1.0, int kNeighbors = 10)
    {
        if (kNeighbors < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(kNeighbors), "kNeighbors must be at least 1.");
        }

        _numOps = MathHelper.GetNumericOperations<T>();
        _beta = beta;
        _kNeighbors = kNeighbors;
        _useBatchDiversity = false;
        _lastMinScore = _numOps.Zero;
        _lastMaxScore = _numOps.Zero;
        _lastMeanScore = _numOps.Zero;
    }

    /// <inheritdoc />
    public string Name => $"DensityWeightedSampling-beta{_beta}-k{_kNeighbors}";

    /// <inheritdoc />
    public bool UseBatchDiversity
    {
        get => _useBatchDiversity;
        set => _useBatchDiversity = value;
    }

    /// <inheritdoc />
    public int[] SelectSamples(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool, int batchSize)
    {
        Guard.NotNull(model);
        Guard.NotNull(unlabeledPool);

        var scores = ComputeInformativenessScores(model, unlabeledPool);
        var numSamples = unlabeledPool.Shape[0];
        batchSize = Math.Min(batchSize, numSamples);

        if (_useBatchDiversity)
        {
            return SelectWithDiversity(scores, unlabeledPool, batchSize);
        }
        else
        {
            return SelectTopScoring(scores, batchSize);
        }
    }

    /// <inheritdoc />
    public Vector<T> ComputeInformativenessScores(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool)
    {
        Guard.NotNull(model);
        Guard.NotNull(unlabeledPool);

        var predictions = model.Predict(unlabeledPool);
        var numSamples = unlabeledPool.Shape[0];
        var numClasses = predictions.Length / numSamples;
        var featureSize = unlabeledPool.Length / numSamples;
        var engine = AiDotNetEngine.Current;

        // Compute uncertainty scores (using entropy)
        var probs = ActiveLearningEngineOps.SoftmaxRows(predictions, numSamples, numClasses);
        var uncertaintyScores = ActiveLearningEngineOps.EntropyRows(probs, ActiveLearningEngineOps.Tolerance<T>(1e-10));

        // Compute density scores
        var densityScores = ComputeDensityScores(unlabeledPool, numSamples, featureSize);

        // Combine: Score = Uncertainty × Density^β
        var weightedDensity = engine.TensorPower(densityScores, _numOps.FromDouble(_beta));
        var scores = engine.TensorMultiply(uncertaintyScores, weightedDensity).ToVector();

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
    /// Computes density scores based on average distance to k nearest neighbors.
    /// Higher density = more neighbors nearby = more representative.
    /// </summary>
    private Tensor<T> ComputeDensityScores(Tensor<T> pool, int numSamples, int featureSize)
    {
        var engine = AiDotNetEngine.Current;
        var k = Math.Min(_kNeighbors, numSamples - 1);
        Tensor<T> avgDist;

        if (k > 0)
        {
            // Average of the k smallest distances to the other samples. The k + 1 smallest
            // distances of a row include the sample's zero distance to itself, so their sum is
            // the sum of the k nearest neighbours.
            var rows = ActiveLearningEngineOps.AsRows(pool, numSamples, featureSize);
            var blockRows = ActiveLearningEngineOps.PairwiseBlockRows(numSamples, featureSize);
            var sums = new List<Tensor<T>>();
            for (int start = 0; start < numSamples; start += blockRows)
            {
                int count = Math.Min(blockRows, numSamples - start);
                var distances = ActiveLearningEngineOps.DistanceBlock(rows, start, count);
                var (nearest, _) = engine.TopK(distances, k + 1, 1, false);
                sums.Add(engine.ReduceSum(nearest, new[] { 1 }, keepDims: false));
            }

            avgDist = engine.TensorDivideScalar(engine.TensorConcatenate(sums.ToArray(), 0), _numOps.FromDouble(k));
        }
        else
        {
            avgDist = new Tensor<T>(new[] { numSamples });
            avgDist.Fill(_numOps.One);
        }

        // Convert to density (inverse of average distance)
        // Add small epsilon to avoid division by zero
        var density = engine.TensorReciprocal(engine.TensorAddScalar(avgDist, ActiveLearningEngineOps.Tolerance<T>(1e-10)));

        // Normalize density scores to [0, 1]
        return NormalizeDensityScores(density);
    }

    /// <summary>
    /// Normalizes density scores to [0, 1] range.
    /// </summary>
    private Tensor<T> NormalizeDensityScores(Tensor<T> scores)
    {
        if (scores.Length == 0)
        {
            return scores;
        }

        var engine = AiDotNetEngine.Current;
        var maxDensity = engine.TensorMaxValue(scores);
        return _numOps.GreaterThan(maxDensity, ActiveLearningEngineOps.Tolerance<T>(1e-10))
            ? engine.TensorDivideScalar(scores, maxDensity)
            : scores;
    }

    /// <summary>
    /// Selects top-scoring samples without diversity consideration.
    /// </summary>
    private static int[] SelectTopScoring(Vector<T> scores, int batchSize)
    {
        return ActiveLearningEngineOps.SelectTopScoring(scores, batchSize);
    }

    /// <summary>
    /// Selects samples considering both density-weighted scores and diversity.
    /// </summary>
    private static int[] SelectWithDiversity(Vector<T> scores, Tensor<T> pool, int batchSize)
    {
        return ActiveLearningEngineOps.SelectWithDiversity(scores, pool, batchSize);
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
