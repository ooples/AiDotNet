using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Implements information density sampling for active learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Information density measures how representative a sample is
/// of the overall data distribution by computing its average similarity to all other samples
/// in the pool. This helps select informative samples that are also typical of the data.</para>
///
/// <para><b>Formula:</b> ID(x) = Uncertainty(x) × [1/|U| × Σ sim(x, x')]^β</para>
/// <para>where sim(x, x') is the similarity between samples (typically cosine or RBF kernel).</para>
///
/// <para><b>Intuition:</b> A sample is information-dense if:</para>
/// <list type="bullet">
/// <item><description>It has high uncertainty (model is unsure).</description></item>
/// <item><description>It is similar to many other samples (representative).</description></item>
/// </list>
///
/// <para><b>Advantages:</b></para>
/// <list type="bullet">
/// <item><description>Balances informativeness and representativeness.</description></item>
/// <item><description>Avoids selecting outliers.</description></item>
/// <item><description>Configurable β parameter for tuning.</description></item>
/// </list>
///
/// <para><b>Reference:</b> McCallum, A. &amp; Nigam, K. (1998). "Employing EM and Pool-Based
/// Active Learning for Text Classification."</para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Optimization)]
[ModelCategory(ModelCategory.InstanceBased)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Employing EM and Pool-Based Active Learning for Text Classification", "https://dl.acm.org/doi/10.5555/645527.757765", Year = 1998, Authors = "Andrew McCallum, Kamal Nigam")]
[ComponentType(ComponentType.ActiveLearner)]
[PipelineStage(PipelineStage.Training)]
public class InformationDensity<T> : IActiveLearningStrategy<T>
{
    /// <summary>
    /// Defines the similarity measure for computing information density.
    /// </summary>
    public enum SimilarityMeasure
    {
        /// <summary>Cosine similarity.</summary>
        Cosine,
        /// <summary>RBF (Gaussian) kernel similarity.</summary>
        RBF,
        /// <summary>Inverse Euclidean distance.</summary>
        InverseEuclidean
    }

    private readonly INumericOperations<T> _numOps;
    private readonly double _beta;
    private readonly double _rbfGamma;
    private readonly SimilarityMeasure _similarityMeasure;
    private bool _useBatchDiversity;
    private T _lastMinScore;
    private T _lastMaxScore;
    private T _lastMeanScore;

    /// <summary>
    /// Initializes a new instance of the InformationDensity class.
    /// </summary>
    /// <param name="beta">Exponent for density weighting (default: 1.0).</param>
    /// <param name="similarityMeasure">Similarity measure to use (default: Cosine).</param>
    /// <param name="rbfGamma">Gamma parameter for RBF kernel (default: 1.0).</param>
    public InformationDensity(
        double beta = 1.0,
        SimilarityMeasure similarityMeasure = SimilarityMeasure.Cosine,
        double rbfGamma = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _beta = beta;
        _similarityMeasure = similarityMeasure;
        _rbfGamma = rbfGamma;
        _useBatchDiversity = false;
        _lastMinScore = _numOps.Zero;
        _lastMaxScore = _numOps.Zero;
        _lastMeanScore = _numOps.Zero;
    }

    /// <inheritdoc />
    public string Name => $"InformationDensity-{_similarityMeasure}-beta{_beta}";

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
        var uncertaintyScores = ActiveLearningEngineOps.EntropyRows(probs, _numOps.FromDouble(1e-10));

        // Compute average similarity to pool
        var avgSimilarities = ComputeAverageSimilarities(unlabeledPool, numSamples, featureSize);

        // Combine: ID(x) = Uncertainty(x) × AvgSimilarity(x)^β
        var weightedSim = engine.TensorPower(avgSimilarities, _numOps.FromDouble(_beta));
        var scores = engine.TensorMultiply(uncertaintyScores, weightedSim).ToVector();

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
    /// Computes average similarity of each sample to all other samples in the pool.
    /// </summary>
    private Tensor<T> ComputeAverageSimilarities(Tensor<T> pool, int numSamples, int featureSize)
    {
        var engine = AiDotNetEngine.Current;
        if (numSamples < 2)
        {
            return new Tensor<T>(new[] { numSamples });
        }

        var rows = ActiveLearningEngineOps.AsRows(pool, numSamples, featureSize);
        var norms = _similarityMeasure is SimilarityMeasure.RBF or SimilarityMeasure.InverseEuclidean
            ? null
            : ActiveLearningEngineOps.RowNorms(rows);
        var blockRows = ActiveLearningEngineOps.PairwiseBlockRows(numSamples, featureSize);
        var totals = new List<Tensor<T>>();

        for (int start = 0; start < numSamples; start += blockRows)
        {
            int count = Math.Min(blockRows, numSamples - start);
            var similarities = ComputeSimilarityBlock(rows, norms, start, count);

            // Exclude each sample's similarity to itself
            var others = engine.TensorMaskedFill(similarities, ActiveLearningEngineOps.SelfMask(start, count, numSamples), _numOps.Zero);
            totals.Add(engine.ReduceSum(others, new[] { 1 }, keepDims: false));
        }

        var total = engine.TensorConcatenate(totals.ToArray(), 0);
        return engine.TensorDivideScalar(total, _numOps.FromDouble(numSamples - 1));
    }

    /// <summary>
    /// Computes the similarity between rows [start, start + count) and every sample using the
    /// configured measure.
    /// </summary>
    /// <returns>A [count, numSamples] tensor.</returns>
    private Tensor<T> ComputeSimilarityBlock(Tensor<T> rows, Tensor<T>? norms, int start, int count)
    {
        switch (_similarityMeasure)
        {
            case SimilarityMeasure.RBF:
                return ComputeRBFSimilarity(rows, start, count);
            case SimilarityMeasure.InverseEuclidean:
                return ComputeInverseEuclidean(rows, start, count);
            default:
                return ComputeCosineSimilarity(rows, norms ?? ActiveLearningEngineOps.RowNorms(rows), start, count);
        }
    }

    /// <summary>
    /// Computes cosine similarity normalized to [0, 1].
    /// </summary>
    private Tensor<T> ComputeCosineSimilarity(Tensor<T> rows, Tensor<T> norms, int start, int count)
    {
        var engine = AiDotNetEngine.Current;
        int n = rows.Shape[0];
        var shape = new[] { count, n };
        var dotProduct = ActiveLearningEngineOps.DotBlock(rows, start, count);

        var blockNorms = engine.TensorSlice(norms, new[] { start }, new[] { count }).Reshape(new[] { count, 1 });
        var normProduct = engine.TensorMultiply(
            engine.TensorBroadcastTo(blockNorms, shape),
            engine.TensorBroadcastTo(norms.Reshape(new[] { 1, n }), shape));

        // Normalize from [-1, 1] to [0, 1]; pairs with a (near) zero norm get similarity 0
        var cosineSim = engine.TensorDivide(dotProduct, normProduct);
        var normalized = engine.TensorDivideScalar(engine.TensorAddScalar(cosineSim, _numOps.One), _numOps.FromDouble(2.0));
        var degenerate = engine.TensorLessThan(normProduct, _numOps.FromDouble(1e-10));
        return engine.TensorWhere(degenerate, new Tensor<T>(shape), normalized);
    }

    /// <summary>
    /// Computes RBF (Gaussian) similarity: exp(-gamma * d^2).
    /// </summary>
    private Tensor<T> ComputeRBFSimilarity(Tensor<T> rows, int start, int count)
    {
        var engine = AiDotNetEngine.Current;
        var dist = ActiveLearningEngineOps.DistanceBlock(rows, start, count);
        return engine.TensorExp(engine.TensorMultiplyScalar(engine.TensorMultiply(dist, dist), _numOps.FromDouble(-_rbfGamma)));
    }

    /// <summary>
    /// Computes inverse Euclidean similarity: 1 / (1 + d).
    /// </summary>
    private Tensor<T> ComputeInverseEuclidean(Tensor<T> rows, int start, int count)
    {
        var engine = AiDotNetEngine.Current;
        var dist = ActiveLearningEngineOps.DistanceBlock(rows, start, count);
        return engine.TensorReciprocal(engine.TensorAddScalar(dist, _numOps.One));
    }

    /// <summary>
    /// Selects top-scoring samples without diversity consideration.
    /// </summary>
    private static int[] SelectTopScoring(Vector<T> scores, int batchSize)
    {
        return ActiveLearningEngineOps.SelectTopScoring(scores, batchSize);
    }

    /// <summary>
    /// Selects samples considering both information density and batch diversity.
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
