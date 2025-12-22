using AiDotNet.ActiveLearning.Config;
using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;

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
public class QueryByCommitteeStrategy<T, TInput, TOutput> : ICommitteeStrategy<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

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

        var scores = new T[unlabeledPool.Count];

        for (int i = 0; i < unlabeledPool.Count; i++)
        {
            var input = unlabeledPool.GetInput(i);
            scores[i] = ComputeDisagreement(input);
        }

        return new Vector<T>(scores);
    }

    /// <inheritdoc/>
    public int[] SelectSamples(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> unlabeledPool,
        int batchSize)
    {
        var scores = ComputeScores(model, unlabeledPool);
        var batchSizeToUse = Math.Min(batchSize, unlabeledPool.Count);

        var indexedScores = scores
            .Select((score, index) => (Index: index, Score: score))
            .OrderByDescending(x => NumOps.ToDouble(x.Score))
            .Take(batchSizeToUse)
            .Select(x => x.Index)
            .ToArray();

        return indexedScores;
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
        var scores = new T[unlabeledPool.Count];

        for (int i = 0; i < unlabeledPool.Count; i++)
        {
            var input = unlabeledPool.GetInput(i);
            var prediction = model.Predict(input);
            var probabilities = ConvertToProbabilities(prediction);
            scores[i] = ComputeEntropy(probabilities);
        }

        return new Vector<T>(scores);
    }

    private T ComputeVoteEntropy(TInput input)
    {
        // Count votes for each class
        var votes = new int[_numClasses];

        foreach (var member in _committee)
        {
            var prediction = member.Predict(input);
            var classIndex = GetPredictedClass(prediction);
            if (classIndex >= 0 && classIndex < _numClasses)
            {
                votes[classIndex]++;
            }
        }

        // Compute entropy of vote distribution
        T entropy = NumOps.Zero;
        T epsilon = NumOps.FromDouble(1e-10);
        T totalVotes = NumOps.FromDouble(_committee.Count);

        for (int c = 0; c < _numClasses; c++)
        {
            if (votes[c] > 0)
            {
                var p = NumOps.Divide(NumOps.FromDouble(votes[c]), totalVotes);
                var pClipped = NumOps.Add(p, epsilon);
                var logP = NumOps.Log(pClipped);
                entropy = NumOps.Subtract(entropy, NumOps.Multiply(p, logP));
            }
        }

        return entropy;
    }

    private T ComputeKLDivergence(TInput input)
    {
        // Collect probability predictions from all committee members
        var allProbabilities = new List<Vector<T>>();

        foreach (var member in _committee)
        {
            var prediction = member.Predict(input);
            var probabilities = ConvertToProbabilities(prediction);
            allProbabilities.Add(probabilities);
        }

        // Compute consensus (average) distribution
        var consensus = ComputeConsensusDistribution(allProbabilities);

        // Compute average KL divergence from consensus
        T totalKL = NumOps.Zero;

        foreach (var probs in allProbabilities)
        {
            var kl = ComputeKL(probs, consensus);
            totalKL = NumOps.Add(totalKL, kl);
        }

        return NumOps.Divide(totalKL, NumOps.FromDouble(allProbabilities.Count));
    }

    private T ComputeSoftVoteVariance(TInput input)
    {
        // Collect probability predictions
        var allProbabilities = new List<Vector<T>>();

        foreach (var member in _committee)
        {
            var prediction = member.Predict(input);
            var probabilities = ConvertToProbabilities(prediction);
            allProbabilities.Add(probabilities);
        }

        // Compute variance across committee for each class
        var consensus = ComputeConsensusDistribution(allProbabilities);
        T totalVariance = NumOps.Zero;

        for (int c = 0; c < consensus.Length; c++)
        {
            T classVariance = NumOps.Zero;
            foreach (var probs in allProbabilities)
            {
                var diff = NumOps.Subtract(probs[c], consensus[c]);
                classVariance = NumOps.Add(classVariance, NumOps.Multiply(diff, diff));
            }
            totalVariance = NumOps.Add(totalVariance, classVariance);
        }

        return NumOps.Divide(totalVariance, NumOps.FromDouble(allProbabilities.Count));
    }

    private Vector<T> ComputeConsensusDistribution(List<Vector<T>> allProbabilities)
    {
        if (allProbabilities.Count == 0)
        {
            return new Vector<T>(_numClasses);
        }

        var consensus = new T[allProbabilities[0].Length];
        for (int i = 0; i < consensus.Length; i++)
        {
            consensus[i] = NumOps.Zero;
        }

        foreach (var probs in allProbabilities)
        {
            for (int i = 0; i < consensus.Length; i++)
            {
                consensus[i] = NumOps.Add(consensus[i], probs[i]);
            }
        }

        var count = NumOps.FromDouble(allProbabilities.Count);
        for (int i = 0; i < consensus.Length; i++)
        {
            consensus[i] = NumOps.Divide(consensus[i], count);
        }

        return new Vector<T>(consensus);
    }

    private T ComputeKL(Vector<T> p, Vector<T> q)
    {
        T kl = NumOps.Zero;
        T epsilon = NumOps.FromDouble(1e-10);

        for (int i = 0; i < p.Length; i++)
        {
            if (NumOps.Compare(p[i], NumOps.Zero) > 0)
            {
                var pClipped = NumOps.Add(p[i], epsilon);
                var qClipped = NumOps.Add(q[i], epsilon);
                var logRatio = NumOps.Log(NumOps.Divide(pClipped, qClipped));
                kl = NumOps.Add(kl, NumOps.Multiply(p[i], logRatio));
            }
        }

        return kl;
    }

    private T ComputeEntropy(Vector<T> probabilities)
    {
        T entropy = NumOps.Zero;
        T epsilon = NumOps.FromDouble(1e-10);

        for (int i = 0; i < probabilities.Length; i++)
        {
            var p = probabilities[i];
            if (NumOps.Compare(p, NumOps.Zero) > 0)
            {
                var pClipped = NumOps.Add(p, epsilon);
                var logP = NumOps.Log(pClipped);
                entropy = NumOps.Subtract(entropy, NumOps.Multiply(p, logP));
            }
        }

        return entropy;
    }

    private int GetPredictedClass(TOutput prediction)
    {
        if (prediction is int intValue) return intValue;
        if (prediction is Vector<T> vectorPred)
        {
            // Return argmax
            int maxIndex = 0;
            T maxValue = vectorPred[0];
            for (int i = 1; i < vectorPred.Length; i++)
            {
                if (NumOps.Compare(vectorPred[i], maxValue) > 0)
                {
                    maxValue = vectorPred[i];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }
        if (prediction is T numValue)
        {
            return NumOps.ToDouble(numValue) >= 0.5 ? 1 : 0;
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
        T maxLogit = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (NumOps.Compare(logits[i], maxLogit) > 0)
            {
                maxLogit = logits[i];
            }
        }

        var expValues = new T[logits.Length];
        T sum = NumOps.Zero;

        for (int i = 0; i < logits.Length; i++)
        {
            var shifted = NumOps.Subtract(logits[i], maxLogit);
            expValues[i] = NumOps.Exp(shifted);
            sum = NumOps.Add(sum, expValues[i]);
        }

        for (int i = 0; i < expValues.Length; i++)
        {
            expValues[i] = NumOps.Divide(expValues[i], sum);
        }

        return new Vector<T>(expValues);
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
