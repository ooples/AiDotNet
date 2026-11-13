using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Implements Query-by-Committee (QBC) for active learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Query-by-Committee uses multiple models (a "committee") to evaluate
/// samples. It selects samples where the committee members disagree the most. The intuition is
/// that disagreement indicates uncertainty in the version space - the region of hypotheses
/// consistent with the labeled data.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>Train multiple models on the same labeled data (with different initializations,
/// architectures, or training procedures).</description></item>
/// <item><description>For each unlabeled sample, have all committee members make predictions.</description></item>
/// <item><description>Measure disagreement using vote entropy or KL divergence.</description></item>
/// <item><description>Select samples with highest disagreement for labeling.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Seung et al., "Query by Committee" (1992). COLT.</para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Query by Committee", "https://doi.org/10.1145/130385.130417", Year = 1992, Authors = "H. Sebastian Seung, Manfred Opper, Haim Sompolinsky")]
[ComponentType(ComponentType.ActiveLearner)]
[PipelineStage(PipelineStage.Training)]
public class QueryByCommittee<T> : IActiveLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly List<IFullModel<T, Tensor<T>, Tensor<T>>> _committee;
    private readonly DisagreementMeasure _measure;
    private bool _useBatchDiversity;
    private T _lastMinScore;
    private T _lastMaxScore;
    private T _lastMeanScore;

    /// <summary>
    /// Defines the disagreement measure to use.
    /// </summary>
    public enum DisagreementMeasure
    {
        /// <summary>Entropy of the vote distribution.</summary>
        VoteEntropy,
        /// <summary>Average KL divergence between committee members and consensus.</summary>
        KLDivergence,
        /// <summary>Variance of predictions across committee members.</summary>
        PredictionVariance
    }

    /// <summary>
    /// Initializes a new instance of the QueryByCommittee class.
    /// </summary>
    /// <param name="committee">The committee of models.</param>
    /// <param name="measure">The disagreement measure to use (default: VoteEntropy).</param>
    public QueryByCommittee(
        IEnumerable<IFullModel<T, Tensor<T>, Tensor<T>>> committee,
        DisagreementMeasure measure = DisagreementMeasure.VoteEntropy)
    {
        Guard.NotNull(committee);

        _numOps = MathHelper.GetNumericOperations<T>();
        _committee = [.. committee];
        _measure = measure;
        _useBatchDiversity = false;
        _lastMinScore = _numOps.Zero;
        _lastMaxScore = _numOps.Zero;
        _lastMeanScore = _numOps.Zero;

        if (_committee.Count < 2)
        {
            throw new ArgumentException("Committee must have at least 2 members.", nameof(committee));
        }
    }

    /// <inheritdoc />
    public string Name => $"QueryByCommittee-{_measure}";

    /// <inheritdoc />
    public bool UseBatchDiversity
    {
        get => _useBatchDiversity;
        set => _useBatchDiversity = value;
    }

    /// <summary>
    /// Gets the committee of models.
    /// </summary>
    public IReadOnlyList<IFullModel<T, Tensor<T>, Tensor<T>>> Committee => _committee;

    /// <inheritdoc />
    public int[] SelectSamples(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool, int batchSize)
    {
        // Note: The 'model' parameter is ignored; we use the committee instead
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
        Guard.NotNull(unlabeledPool);

        // Get predictions from all committee members
        var allPredictions = new List<Tensor<T>>();
        foreach (var member in _committee)
        {
            var preds = member.Predict(unlabeledPool);
            allPredictions.Add(preds);
        }

        var numSamples = unlabeledPool.Shape[0];
        var numClasses = allPredictions[0].Length / numSamples;
        var scores = ComputeDisagreement(allPredictions, numSamples, numClasses).ToVector();

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
            ["CommitteeSize"] = _numOps.FromDouble(_committee.Count)
        };
    }

    /// <summary>
    /// Computes the disagreement of every sample based on the configured measure.
    /// </summary>
    private Tensor<T> ComputeDisagreement(List<Tensor<T>> allPredictions, int numSamples, int numClasses)
    {
        return _measure switch
        {
            DisagreementMeasure.VoteEntropy => ComputeVoteEntropy(allPredictions, numSamples, numClasses),
            DisagreementMeasure.KLDivergence => ComputeAverageKL(allPredictions, numSamples, numClasses),
            DisagreementMeasure.PredictionVariance => ComputePredictionVariance(allPredictions, numSamples, numClasses),
            _ => ComputeVoteEntropy(allPredictions, numSamples, numClasses)
        };
    }

    /// <summary>
    /// Computes vote entropy: entropy of the vote distribution across committee.
    /// </summary>
    private Tensor<T> ComputeVoteEntropy(List<Tensor<T>> allPredictions, int numSamples, int numClasses)
    {
        var engine = AiDotNetEngine.Current;

        // Count votes for each class (each member votes for its first highest logit)
        var votes = new Tensor<T>(new[] { numSamples, numClasses });
        foreach (var preds in allPredictions)
        {
            var maxClass = engine.TensorArgMax(ActiveLearningEngineOps.AsRows(preds, numSamples, numClasses), 1);
            votes = engine.TensorAdd(votes, engine.TensorOneHot<T>(maxClass, numClasses));
        }

        // Compute entropy of vote distribution over classes with p > epsilon
        var p = engine.TensorDivideScalar(votes, _numOps.FromDouble(_committee.Count));
        var terms = engine.TensorMultiply(p, engine.TensorLog(p));
        var counted = engine.TensorGreaterThan(p, ActiveLearningEngineOps.Tolerance<T>(1e-10));
        var kept = engine.TensorWhere(counted, terms, new Tensor<T>(new[] { numSamples, numClasses }));
        return engine.TensorNegate(engine.ReduceSum(kept, new[] { 1 }, keepDims: false));
    }

    /// <summary>
    /// Computes average KL divergence between each member and the consensus.
    /// </summary>
    private Tensor<T> ComputeAverageKL(List<Tensor<T>> allPredictions, int numSamples, int numClasses)
    {
        var engine = AiDotNetEngine.Current;
        var epsilon = ActiveLearningEngineOps.Tolerance<T>(1e-10);
        var committeeSize = _numOps.FromDouble(_committee.Count);
        var memberProbs = SoftmaxAll(allPredictions, numSamples, numClasses);

        // Compute consensus (average) probability distribution
        var consensus = engine.TensorDivideScalar(Sum(memberProbs, numSamples, numClasses), committeeSize);
        var logQ = engine.TensorLog(engine.TensorAddScalar(consensus, epsilon));

        // Compute average KL divergence
        var totalKL = new Tensor<T>(new[] { numSamples });
        foreach (var p in memberProbs)
        {
            var logP = engine.TensorLog(engine.TensorAddScalar(p, epsilon));
            var terms = engine.TensorMultiply(p, engine.TensorSubtract(logP, logQ));
            totalKL = engine.TensorAdd(totalKL, engine.ReduceSum(terms, new[] { 1 }, keepDims: false));
        }

        return engine.TensorDivideScalar(totalKL, committeeSize);
    }

    /// <summary>
    /// Computes variance of predictions across committee members.
    /// </summary>
    private Tensor<T> ComputePredictionVariance(List<Tensor<T>> allPredictions, int numSamples, int numClasses)
    {
        var engine = AiDotNetEngine.Current;
        var committeeSize = _numOps.FromDouble(_committee.Count);
        var memberProbs = SoftmaxAll(allPredictions, numSamples, numClasses);

        // Compute mean
        var mean = engine.TensorDivideScalar(Sum(memberProbs, numSamples, numClasses), committeeSize);

        // Compute variance per class, then total over classes
        var squaredDeviation = new Tensor<T>(new[] { numSamples, numClasses });
        foreach (var p in memberProbs)
        {
            var diff = engine.TensorSubtract(p, mean);
            squaredDeviation = engine.TensorAdd(squaredDeviation, engine.TensorMultiply(diff, diff));
        }

        var variance = engine.TensorDivideScalar(squaredDeviation, committeeSize);
        return engine.ReduceSum(variance, new[] { 1 }, keepDims: false);
    }

    /// <summary>
    /// Softmax probabilities of every committee member, each [numSamples, numClasses].
    /// </summary>
    private static List<Tensor<T>> SoftmaxAll(List<Tensor<T>> allPredictions, int numSamples, int numClasses)
    {
        var probs = new List<Tensor<T>>(allPredictions.Count);
        foreach (var preds in allPredictions)
        {
            probs.Add(ActiveLearningEngineOps.SoftmaxRows(preds, numSamples, numClasses));
        }

        return probs;
    }

    /// <summary>
    /// Element-wise sum of [numSamples, numClasses] tensors.
    /// </summary>
    private static Tensor<T> Sum(List<Tensor<T>> tensors, int numSamples, int numClasses)
    {
        var engine = AiDotNetEngine.Current;
        var sum = new Tensor<T>(new[] { numSamples, numClasses });
        foreach (var tensor in tensors)
        {
            sum = engine.TensorAdd(sum, tensor);
        }

        return sum;
    }

    /// <summary>
    /// Selects top-scoring samples.
    /// </summary>
    private static int[] SelectTopScoring(Vector<T> scores, int batchSize)
    {
        return ActiveLearningEngineOps.SelectTopScoring(scores, batchSize);
    }

    /// <summary>
    /// Selects samples considering both committee disagreement score and diversity.
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
