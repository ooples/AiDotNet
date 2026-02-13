using AiDotNet.ActiveLearning.Interfaces;
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
        var scores = new Vector<T>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            scores[i] = ComputeDisagreement(allPredictions, i, numClasses);
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
            ["CommitteeSize"] = _numOps.FromDouble(_committee.Count)
        };
    }

    /// <summary>
    /// Computes disagreement for a single sample based on the configured measure.
    /// </summary>
    private T ComputeDisagreement(List<Tensor<T>> allPredictions, int sampleIndex, int numClasses)
    {
        return _measure switch
        {
            DisagreementMeasure.VoteEntropy => ComputeVoteEntropy(allPredictions, sampleIndex, numClasses),
            DisagreementMeasure.KLDivergence => ComputeAverageKL(allPredictions, sampleIndex, numClasses),
            DisagreementMeasure.PredictionVariance => ComputePredictionVariance(allPredictions, sampleIndex, numClasses),
            _ => ComputeVoteEntropy(allPredictions, sampleIndex, numClasses)
        };
    }

    /// <summary>
    /// Computes vote entropy: entropy of the vote distribution across committee.
    /// </summary>
    private T ComputeVoteEntropy(List<Tensor<T>> allPredictions, int sampleIndex, int numClasses)
    {
        var votes = new int[numClasses];
        var startIdx = sampleIndex * numClasses;

        // Count votes for each class
        foreach (var preds in allPredictions)
        {
            var maxClass = 0;
            var maxVal = preds[startIdx];
            for (int c = 1; c < numClasses; c++)
            {
                if (_numOps.GreaterThan(preds[startIdx + c], maxVal))
                {
                    maxVal = preds[startIdx + c];
                    maxClass = c;
                }
            }
            votes[maxClass]++;
        }

        // Compute entropy of vote distribution
        var entropy = _numOps.Zero;
        var numMembers = (double)_committee.Count;
        var epsilon = 1e-10;

        for (int c = 0; c < numClasses; c++)
        {
            var p = votes[c] / numMembers;
            if (p > epsilon)
            {
                var term = p * Math.Log(p);
                entropy = _numOps.Subtract(entropy, _numOps.FromDouble(term));
            }
        }

        return entropy;
    }

    /// <summary>
    /// Computes average KL divergence between each member and the consensus.
    /// </summary>
    private T ComputeAverageKL(List<Tensor<T>> allPredictions, int sampleIndex, int numClasses)
    {
        var startIdx = sampleIndex * numClasses;
        var consensus = new Vector<T>(numClasses);
        var epsilon = _numOps.FromDouble(1e-10);

        // Compute consensus (average) probability distribution
        for (int c = 0; c < numClasses; c++)
        {
            var sum = _numOps.Zero;
            foreach (var preds in allPredictions)
            {
                sum = _numOps.Add(sum, GetSoftmaxProb(preds, startIdx, c, numClasses));
            }
            consensus[c] = _numOps.Divide(sum, _numOps.FromDouble(_committee.Count));
        }

        // Compute average KL divergence
        var totalKL = _numOps.Zero;
        foreach (var preds in allPredictions)
        {
            var kl = _numOps.Zero;
            for (int c = 0; c < numClasses; c++)
            {
                var p = GetSoftmaxProb(preds, startIdx, c, numClasses);
                var q = _numOps.Add(consensus[c], epsilon);
                var pPlusEps = _numOps.Add(p, epsilon);

                var logP = _numOps.FromDouble(Math.Log(_numOps.ToDouble(pPlusEps)));
                var logQ = _numOps.FromDouble(Math.Log(_numOps.ToDouble(q)));
                var term = _numOps.Multiply(p, _numOps.Subtract(logP, logQ));
                kl = _numOps.Add(kl, term);
            }
            totalKL = _numOps.Add(totalKL, kl);
        }

        return _numOps.Divide(totalKL, _numOps.FromDouble(_committee.Count));
    }

    /// <summary>
    /// Computes variance of predictions across committee members.
    /// </summary>
    private T ComputePredictionVariance(List<Tensor<T>> allPredictions, int sampleIndex, int numClasses)
    {
        var startIdx = sampleIndex * numClasses;
        var totalVariance = _numOps.Zero;

        for (int c = 0; c < numClasses; c++)
        {
            // Compute mean
            var mean = _numOps.Zero;
            foreach (var preds in allPredictions)
            {
                mean = _numOps.Add(mean, GetSoftmaxProb(preds, startIdx, c, numClasses));
            }
            mean = _numOps.Divide(mean, _numOps.FromDouble(_committee.Count));

            // Compute variance
            var variance = _numOps.Zero;
            foreach (var preds in allPredictions)
            {
                var p = GetSoftmaxProb(preds, startIdx, c, numClasses);
                var diff = _numOps.Subtract(p, mean);
                variance = _numOps.Add(variance, _numOps.Multiply(diff, diff));
            }
            variance = _numOps.Divide(variance, _numOps.FromDouble(_committee.Count));
            totalVariance = _numOps.Add(totalVariance, variance);
        }

        return totalVariance;
    }

    /// <summary>
    /// Gets softmax probability for a specific class.
    /// </summary>
    private T GetSoftmaxProb(Tensor<T> logits, int startIdx, int classIdx, int numClasses)
    {
        var maxLogit = _numOps.MinValue;
        for (int c = 0; c < numClasses; c++)
        {
            if (_numOps.GreaterThan(logits[startIdx + c], maxLogit))
            {
                maxLogit = logits[startIdx + c];
            }
        }

        var expSum = _numOps.Zero;
        var expTarget = _numOps.Zero;
        for (int c = 0; c < numClasses; c++)
        {
            var shifted = _numOps.Subtract(logits[startIdx + c], maxLogit);
            var expVal = _numOps.FromDouble(Math.Exp(_numOps.ToDouble(shifted)));
            expSum = _numOps.Add(expSum, expVal);
            if (c == classIdx)
            {
                expTarget = expVal;
            }
        }

        return _numOps.Divide(expTarget, expSum);
    }

    /// <summary>
    /// Selects top-scoring samples.
    /// </summary>
    private int[] SelectTopScoring(Vector<T> scores, int batchSize)
    {
        var indexedScores = new List<(int Index, T Score)>();
        for (int i = 0; i < scores.Length; i++)
        {
            indexedScores.Add((i, scores[i]));
        }

        return indexedScores
            .OrderByDescending(x => _numOps.ToDouble(x.Score))
            .Take(batchSize)
            .Select(x => x.Index)
            .ToArray();
    }

    /// <summary>
    /// Selects samples considering both committee disagreement score and diversity.
    /// </summary>
    private int[] SelectWithDiversity(Vector<T> scores, Tensor<T> pool, int batchSize)
    {
        var selected = new List<int>();
        var remaining = new HashSet<int>(Enumerable.Range(0, scores.Length));
        var featureSize = pool.Length / pool.Shape[0];

        while (selected.Count < batchSize && remaining.Count > 0)
        {
            var best = -1;
            var bestCombinedScore = _numOps.MinValue;

            foreach (var idx in remaining)
            {
                var disagreementScore = scores[idx];
                var diversityScore = selected.Count == 0
                    ? _numOps.One
                    : ComputeMinDistanceToSelected(pool, idx, selected, featureSize);

                var combinedScore = _numOps.Multiply(disagreementScore, diversityScore);

                if (_numOps.GreaterThan(combinedScore, bestCombinedScore))
                {
                    bestCombinedScore = combinedScore;
                    best = idx;
                }
            }

            if (best >= 0)
            {
                selected.Add(best);
                remaining.Remove(best);
            }
        }

        return [.. selected];
    }

    /// <summary>
    /// Computes minimum distance from a sample to already selected samples.
    /// </summary>
    private T ComputeMinDistanceToSelected(Tensor<T> pool, int sampleIdx, List<int> selected, int featureSize)
    {
        var minDist = _numOps.MaxValue;

        foreach (var selIdx in selected)
        {
            var dist = ComputeEuclideanDistance(pool, sampleIdx, selIdx, featureSize);
            if (_numOps.LessThan(dist, minDist))
            {
                minDist = dist;
            }
        }

        return minDist;
    }

    /// <summary>
    /// Computes Euclidean distance between two samples.
    /// </summary>
    private T ComputeEuclideanDistance(Tensor<T> pool, int idx1, int idx2, int featureSize)
    {
        var sumSquared = _numOps.Zero;
        var start1 = idx1 * featureSize;
        var start2 = idx2 * featureSize;

        for (int i = 0; i < featureSize; i++)
        {
            var diff = _numOps.Subtract(pool[start1 + i], pool[start2 + i]);
            var squared = _numOps.Multiply(diff, diff);
            sumSquared = _numOps.Add(sumSquared, squared);
        }

        return _numOps.FromDouble(Math.Sqrt(_numOps.ToDouble(sumSquared)));
    }

    /// <summary>
    /// Updates selection statistics.
    /// </summary>
    private void UpdateStatistics(Vector<T> scores)
    {
        if (scores.Length == 0) return;

        _lastMinScore = scores[0];
        _lastMaxScore = scores[0];
        var sum = _numOps.Zero;

        for (int i = 0; i < scores.Length; i++)
        {
            if (_numOps.LessThan(scores[i], _lastMinScore))
                _lastMinScore = scores[i];
            if (_numOps.GreaterThan(scores[i], _lastMaxScore))
                _lastMaxScore = scores[i];
            sum = _numOps.Add(sum, scores[i]);
        }

        _lastMeanScore = _numOps.Divide(sum, _numOps.FromDouble(scores.Length));
    }
}
