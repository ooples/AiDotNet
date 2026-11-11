using AiDotNet.ActiveLearning.Config;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ActiveLearning.Strategies.Committee;

/// <summary>
/// Query By Committee (QBC) strategy for active learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> QBC uses a "committee" of diverse models to identify samples
/// where the models disagree most. High disagreement indicates uncertainty about the true label.</para>
///
/// <para><b>How QBC Works:</b></para>
/// <list type="number">
/// <item><description>Train multiple models (committee) on the same data</description></item>
/// <item><description>For each unlabeled sample, collect predictions from all committee members</description></item>
/// <item><description>Measure disagreement using vote entropy or KL divergence</description></item>
/// <item><description>Select samples with highest disagreement for labeling</description></item>
/// </list>
///
/// <para><b>Disagreement Measures:</b></para>
/// <list type="bullet">
/// <item><description><b>Vote Entropy:</b> Entropy of the vote distribution</description></item>
/// <item><description><b>KL Divergence:</b> Average divergence from consensus</description></item>
/// <item><description><b>Soft Vote Variance:</b> Variance of probability predictions</description></item>
/// </list>
///
/// <para><b>When to Use:</b></para>
/// <list type="bullet">
/// <item><description>Can train multiple diverse models</description></item>
/// <item><description>Want to capture model uncertainty through ensemble disagreement</description></item>
/// <item><description>Classification problems with discrete predictions</description></item>
/// </list>
///
/// <para><b>Reference:</b> Seung et al. "Query by Committee" (1992)</para>
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
public class QueryByCommitteeStrategy<T, TInput, TOutput> : ICommitteeStrategy<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static readonly T Epsilon = NumOps.FromDouble(1e-10);

    private readonly List<IFullModel<T, TInput, TOutput>> _committee;
    private readonly CommitteeDisagreementMeasure _disagreementMeasure;
    private readonly int _numClasses;
    private readonly ActiveLearnerConfig<T>? _config;

    /// <inheritdoc/>
    public string Name => $"QueryByCommittee ({_disagreementMeasure})";

    /// <inheritdoc/>
    public string Description => _disagreementMeasure switch
    {
        CommitteeDisagreementMeasure.VoteEntropy => "Selects samples with highest vote entropy among committee members",
        CommitteeDisagreementMeasure.KLDivergence => "Selects samples with highest KL divergence from consensus",
        CommitteeDisagreementMeasure.SoftVoteVariance => "Selects samples with highest variance in soft predictions",
        _ => "Selects samples where committee members disagree most"
    };

    /// <inheritdoc/>
    public IReadOnlyList<IFullModel<T, TInput, TOutput>> Committee => _committee.AsReadOnly();

    /// <summary>
    /// Initializes a new QBC strategy with an empty committee.
    /// </summary>
    /// <param name="numClasses">Number of output classes.</param>
    public QueryByCommitteeStrategy(int numClasses)
        : this([], CommitteeDisagreementMeasure.VoteEntropy, numClasses, null)
    {
    }

    /// <summary>
    /// Initializes a new QBC strategy with a pre-built committee.
    /// </summary>
    /// <param name="committee">The committee of models.</param>
    /// <param name="disagreementMeasure">The disagreement measure to use.</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <param name="config">Optional configuration.</param>
    public QueryByCommitteeStrategy(
        IEnumerable<IFullModel<T, TInput, TOutput>> committee,
        CommitteeDisagreementMeasure disagreementMeasure,
        int numClasses,
        ActiveLearnerConfig<T>? config = null)
    {
        _committee = committee.ToList();
        _disagreementMeasure = disagreementMeasure;
        _numClasses = numClasses > 0 ? numClasses : 2;
        _config = config;
    }

    /// <inheritdoc/>
    public Vector<T> ComputeScores(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> unlabeledPool)
    {
        if (_committee.Count == 0)
        {
            // If no committee, use the single model with noise for diversity
            return ComputeScoresWithSingleModel(model, unlabeledPool);
        }

        var scores = new List<T>(unlabeledPool.Count);

        for (int i = 0; i < unlabeledPool.Count; i++)
        {
            var input = unlabeledPool.GetInput(i);
            scores.Add(ComputeDisagreement(input));
        }

        return Vector<T>.FromList(scores);
    }

    /// <inheritdoc/>
    public int[] SelectSamples(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> unlabeledPool,
        int batchSize)
    {
        var scores = ComputeScores(model, unlabeledPool);
        var batchSizeToUse = Math.Min(batchSize, unlabeledPool.Count);

        return ActiveLearningEngineOps.SelectTopScoring(scores, batchSizeToUse);
    }

    /// <inheritdoc/>
    public T ComputeDisagreement(TInput input)
    {
        if (_committee.Count < 2)
        {
            return NumOps.Zero;
        }

        return _disagreementMeasure switch
        {
            CommitteeDisagreementMeasure.VoteEntropy => ComputeVoteEntropy(input),
            CommitteeDisagreementMeasure.KLDivergence => ComputeKLDivergence(input),
            CommitteeDisagreementMeasure.SoftVoteVariance => ComputeSoftVoteVariance(input),
            _ => ComputeVoteEntropy(input)
        };
    }

    /// <inheritdoc/>
    public void AddCommitteeMember(IFullModel<T, TInput, TOutput> model)
    {
        _committee.Add(model);
    }

    /// <inheritdoc/>
    public void RemoveCommitteeMember(IFullModel<T, TInput, TOutput> model)
    {
        _committee.Remove(model);
    }

    /// <inheritdoc/>
    public void UpdateCommittee(IDataset<T, TInput, TOutput> trainingData)
    {
        // Retrain all committee members with the updated training data
        foreach (var member in _committee)
        {
            if (member is ITrainableModel<T, TInput, TOutput> trainable)
            {
                trainable.Train(trainingData);
            }
        }
    }

    /// <inheritdoc/>
    public void UpdateState(int[] newlyLabeledIndices, TOutput[] labels)
    {
        // QBC may need to retrain committee members - handled externally
    }

    /// <inheritdoc/>
    public void Reset()
    {
        _committee.Clear();
    }

    #region Private Methods

    private Vector<T> ComputeScoresWithSingleModel(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> unlabeledPool)
    {
        // Fallback: use entropy-based uncertainty from single model
        var scores = new List<T>(unlabeledPool.Count);

        for (int i = 0; i < unlabeledPool.Count; i++)
        {
            var input = unlabeledPool.GetInput(i);
            var prediction = model.Predict(input);
            var probabilities = ConvertToProbabilities(prediction);
            scores.Add(ComputeEntropy(Tensor<T>.FromVector(probabilities)));
        }

        return Vector<T>.FromList(scores);
    }

    private T ComputeVoteEntropy(TInput input)
    {
        // Collect the valid class votes
        var votedClasses = new List<int>(_committee.Count);

        foreach (var member in _committee)
        {
            var prediction = member.Predict(input);
            var classIndex = GetPredictedClass(prediction);
            if (classIndex >= 0 && classIndex < _numClasses)
            {
                votedClasses.Add(classIndex);
            }
        }

        if (votedClasses.Count == 0)
        {
            return NumOps.Zero;
        }

        // Count votes for each class
        var engine = AiDotNetEngine.Current;
        var ones = new Tensor<T>(new[] { votedClasses.Count });
        ones.Fill(NumOps.One);
        var votes = engine.TensorIndexAdd(
            new Tensor<T>(new[] { _numClasses }), 0, new Tensor<int>(votedClasses.ToArray(), new[] { votedClasses.Count }), ones);

        // Compute entropy of vote distribution
        var voteShares = engine.TensorDivideScalar(votes, NumOps.FromDouble(_committee.Count));
        return ComputeEntropy(voteShares);
    }

    private T ComputeKLDivergence(TInput input)
    {
        // Collect probability predictions from all committee members
        var allProbabilities = new List<Vector<T>>();

        foreach (var member in _committee)
        {
            var prediction = member.Predict(input);
            allProbabilities.Add(ConvertToProbabilities(prediction));
        }

        // Compute consensus (average) distribution
        var engine = AiDotNetEngine.Current;
        var probabilities = ActiveLearningEngineOps.StackVectors(allProbabilities);
        var consensus = ComputeConsensusDistribution(probabilities);

        // Compute average KL divergence from consensus
        var shape = probabilities.Shape.ToArray();
        var q = engine.TensorBroadcastTo(consensus.Reshape(new[] { 1, shape[1] }), shape);
        var ratio = engine.TensorDivide(engine.TensorAddScalar(probabilities, Epsilon), engine.TensorAddScalar(q, Epsilon));
        var terms = PositiveWhere(probabilities, engine.TensorMultiply(probabilities, engine.TensorLog(ratio)));
        T totalKL = engine.TensorSum(engine.ReduceSum(terms, new[] { 1 }, keepDims: false));

        return NumOps.Divide(totalKL, NumOps.FromDouble(allProbabilities.Count));
    }

    private T ComputeSoftVoteVariance(TInput input)
    {
        // Collect probability predictions
        var allProbabilities = new List<Vector<T>>();

        foreach (var member in _committee)
        {
            var prediction = member.Predict(input);
            allProbabilities.Add(ConvertToProbabilities(prediction));
        }

        // Compute variance across committee for each class
        var engine = AiDotNetEngine.Current;
        var probabilities = ActiveLearningEngineOps.StackVectors(allProbabilities);
        var consensus = ComputeConsensusDistribution(probabilities);
        var shape = probabilities.Shape.ToArray();
        var diff = engine.TensorSubtract(probabilities, engine.TensorBroadcastTo(consensus.Reshape(new[] { 1, shape[1] }), shape));
        var classVariance = engine.ReduceSum(engine.TensorMultiply(diff, diff), new[] { 0 }, keepDims: false);
        T totalVariance = engine.TensorSum(classVariance);

        return NumOps.Divide(totalVariance, NumOps.FromDouble(allProbabilities.Count));
    }

    /// <summary>
    /// Mean of the [members, classes] probability rows over the committee members.
    /// </summary>
    private static Tensor<T> ComputeConsensusDistribution(Tensor<T> probabilities)
    {
        var engine = AiDotNetEngine.Current;
        var sum = engine.ReduceSum(probabilities, new[] { 0 }, keepDims: false);
        return engine.TensorDivideScalar(sum, NumOps.FromDouble(probabilities.Shape[0]));
    }

    /// <summary>
    /// Entropy -sum(p * log(p + eps)) over the entries with p &gt; 0.
    /// </summary>
    private static T ComputeEntropy(Tensor<T> probabilities)
    {
        var engine = AiDotNetEngine.Current;
        var terms = engine.TensorMultiply(probabilities, engine.TensorLog(engine.TensorAddScalar(probabilities, Epsilon)));
        return NumOps.Negate(engine.TensorSum(PositiveWhere(probabilities, terms)));
    }

    /// <summary>
    /// <paramref name="terms"/> where <paramref name="probabilities"/> &gt; 0, zero elsewhere.
    /// </summary>
    private static Tensor<T> PositiveWhere(Tensor<T> probabilities, Tensor<T> terms)
    {
        var engine = AiDotNetEngine.Current;
        var positive = engine.TensorGreaterThan(probabilities, NumOps.Zero);
        return engine.TensorWhere(positive, terms, new Tensor<T>(terms.Shape.ToArray()));
    }

    private int GetPredictedClass(TOutput prediction)
    {
        if (prediction is int intValue) return intValue;
        if (prediction is Vector<T> vectorPred)
        {
            // Return argmax (first maximum)
            if (vectorPred.Length == 0)
            {
                throw new ArgumentException("Prediction vector must not be empty.", nameof(prediction));
            }

            var row = Tensor<T>.FromVector(vectorPred).Reshape(new[] { 1, vectorPred.Length });
            return AiDotNetEngine.Current.TensorArgMax(row, 1).GetFlat(0);
        }
        if (prediction is T numValue)
        {
            return NumOps.GreaterThanOrEquals(numValue, NumOps.FromDouble(0.5)) ? 1 : 0;
        }
        return 0;
    }

    private Vector<T> ConvertToProbabilities(TOutput prediction)
    {
        if (prediction is Vector<T> vectorPred)
        {
            return Softmax(vectorPred);
        }

        // For scalar predictions, create binary probability
        var value = ConvertToNumeric(prediction);
        var p = Sigmoid(value);
        var oneMinusP = NumOps.Subtract(NumOps.One, p);
        return new Vector<T>(new[] { oneMinusP, p });
    }

    private Vector<T> Softmax(Vector<T> logits)
    {
        if (logits.Length == 0)
        {
            throw new ArgumentException("Logits must not be empty.", nameof(logits));
        }

        var engine = AiDotNetEngine.Current;
        var values = Tensor<T>.FromVector(logits);
        var expValues = engine.TensorExp(engine.TensorSubtractScalar(values, engine.TensorMaxValue(values)));
        return engine.TensorDivideScalar(expValues, engine.TensorSum(expValues)).ToVector();
    }

    private T Sigmoid(T x)
    {
        var negX = NumOps.Negate(x);
        var expNegX = NumOps.Exp(negX);
        var denominator = NumOps.Add(NumOps.One, expNegX);
        return NumOps.Divide(NumOps.One, denominator);
    }

    private T ConvertToNumeric(TOutput output)
    {
        if (output is T typedValue) return typedValue;
        if (output is double doubleValue) return NumOps.FromDouble(doubleValue);
        if (output is float floatValue) return NumOps.FromDouble(floatValue);
        if (output is int intValue) return NumOps.FromDouble(intValue);
        return NumOps.Zero;
    }

    #endregion
}

/// <summary>
/// Disagreement measures for Query By Committee.
/// </summary>
public enum CommitteeDisagreementMeasure
{
    /// <summary>
    /// Entropy of the hard vote distribution.
    /// </summary>
    VoteEntropy,

    /// <summary>
    /// Average KL divergence from consensus distribution.
    /// </summary>
    KLDivergence,

    /// <summary>
    /// Variance of soft probability predictions.
    /// </summary>
    SoftVoteVariance
}
