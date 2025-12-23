using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Implements uncertainty sampling for active learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Uncertainty sampling is one of the simplest and most popular
/// active learning strategies. It selects samples where the model is least confident about
/// its predictions. The intuition is that uncertain samples are near the decision boundary
/// and provide the most information for learning.</para>
///
/// <para><b>Uncertainty measures:</b></para>
/// <list type="bullet">
/// <item><description><b>Least Confidence:</b> 1 - max(probabilities). Selects samples where
/// the top prediction has low probability.</description></item>
/// <item><description><b>Margin Sampling:</b> P(1st) - P(2nd). Selects samples where the gap
/// between the top two predictions is small.</description></item>
/// <item><description><b>Entropy:</b> -Σ p*log(p). Selects samples where the probability
/// distribution is spread out (high entropy = high uncertainty).</description></item>
/// </list>
///
/// <para><b>Reference:</b> Lewis and Gale, "A Sequential Algorithm for Training Text Classifiers" (1994).</para>
/// </remarks>
public class UncertaintySampling<T> : IActiveLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly UncertaintyMeasure _measure;
    private bool _useBatchDiversity;
    private T _lastMinScore;
    private T _lastMaxScore;
    private T _lastMeanScore;

    /// <summary>
    /// Defines the uncertainty measure to use.
    /// </summary>
    public enum UncertaintyMeasure
    {
        /// <summary>1 - max(probabilities)</summary>
        LeastConfidence,
        /// <summary>P(1st) - P(2nd)</summary>
        MarginSampling,
        /// <summary>-Σ p*log(p)</summary>
        Entropy
    }

    /// <summary>
    /// Initializes a new instance of the UncertaintySampling class.
    /// </summary>
    /// <param name="measure">The uncertainty measure to use (default: Entropy).</param>
    public UncertaintySampling(UncertaintyMeasure measure = UncertaintyMeasure.Entropy)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _measure = measure;
        _useBatchDiversity = false;
        _lastMinScore = _numOps.Zero;
        _lastMaxScore = _numOps.Zero;
        _lastMeanScore = _numOps.Zero;
    }

    /// <inheritdoc />
    public string Name => $"UncertaintySampling-{_measure}";

    /// <inheritdoc />
    public bool UseBatchDiversity
    {
        get => _useBatchDiversity;
        set => _useBatchDiversity = value;
    }

    /// <inheritdoc />
    public int[] SelectSamples(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool, int batchSize)
    {
        _ = model ?? throw new ArgumentNullException(nameof(model));
        _ = unlabeledPool ?? throw new ArgumentNullException(nameof(unlabeledPool));

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
        _ = model ?? throw new ArgumentNullException(nameof(model));
        _ = unlabeledPool ?? throw new ArgumentNullException(nameof(unlabeledPool));

        var predictions = model.Predict(unlabeledPool);
        var numSamples = unlabeledPool.Shape[0];
        var numClasses = predictions.Length / numSamples;
        var scores = new Vector<T>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            var probs = ExtractProbabilities(predictions, i, numClasses);
            scores[i] = ComputeUncertainty(probs);
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
            ["MeanScore"] = _lastMeanScore
        };
    }

    /// <summary>
    /// Computes uncertainty for a single sample based on the configured measure.
    /// </summary>
    private T ComputeUncertainty(Vector<T> probabilities)
    {
        return _measure switch
        {
            UncertaintyMeasure.LeastConfidence => ComputeLeastConfidence(probabilities),
            UncertaintyMeasure.MarginSampling => ComputeMarginSampling(probabilities),
            UncertaintyMeasure.Entropy => ComputeEntropy(probabilities),
            _ => ComputeEntropy(probabilities)
        };
    }

    /// <summary>
    /// Computes least confidence: 1 - max(probabilities).
    /// </summary>
    private T ComputeLeastConfidence(Vector<T> probabilities)
    {
        var maxProb = _numOps.Zero;
        for (int i = 0; i < probabilities.Length; i++)
        {
            if (_numOps.GreaterThan(probabilities[i], maxProb))
            {
                maxProb = probabilities[i];
            }
        }
        return _numOps.Subtract(_numOps.One, maxProb);
    }

    /// <summary>
    /// Computes margin sampling: 1 - (P(1st) - P(2nd)).
    /// </summary>
    private T ComputeMarginSampling(Vector<T> probabilities)
    {
        var first = _numOps.Zero;
        var second = _numOps.Zero;

        for (int i = 0; i < probabilities.Length; i++)
        {
            if (_numOps.GreaterThan(probabilities[i], first))
            {
                second = first;
                first = probabilities[i];
            }
            else if (_numOps.GreaterThan(probabilities[i], second))
            {
                second = probabilities[i];
            }
        }

        var margin = _numOps.Subtract(first, second);
        return _numOps.Subtract(_numOps.One, margin);
    }

    /// <summary>
    /// Computes entropy: -Σ p*log(p).
    /// </summary>
    private T ComputeEntropy(Vector<T> probabilities)
    {
        var entropy = _numOps.Zero;
        var epsilon = _numOps.FromDouble(1e-10);

        for (int i = 0; i < probabilities.Length; i++)
        {
            var p = _numOps.Add(probabilities[i], epsilon);
            var logP = _numOps.FromDouble(Math.Log(_numOps.ToDouble(p)));
            var term = _numOps.Multiply(p, logP);
            entropy = _numOps.Subtract(entropy, term);
        }

        return entropy;
    }

    /// <summary>
    /// Extracts probabilities for a single sample from batch predictions.
    /// </summary>
    private Vector<T> ExtractProbabilities(Tensor<T> predictions, int sampleIndex, int numClasses)
    {
        var probs = new Vector<T>(numClasses);
        var startIdx = sampleIndex * numClasses;

        // Apply softmax to get probabilities
        var maxLogit = _numOps.MinValue;
        for (int c = 0; c < numClasses; c++)
        {
            if (_numOps.GreaterThan(predictions[startIdx + c], maxLogit))
            {
                maxLogit = predictions[startIdx + c];
            }
        }

        var expSum = _numOps.Zero;
        for (int c = 0; c < numClasses; c++)
        {
            var shifted = _numOps.Subtract(predictions[startIdx + c], maxLogit);
            var expVal = _numOps.FromDouble(Math.Exp(_numOps.ToDouble(shifted)));
            probs[c] = expVal;
            expSum = _numOps.Add(expSum, expVal);
        }

        for (int c = 0; c < numClasses; c++)
        {
            probs[c] = _numOps.Divide(probs[c], expSum);
        }

        return probs;
    }

    /// <summary>
    /// Selects top-scoring samples without diversity consideration.
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
    /// Selects samples considering both uncertainty and diversity.
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
                var uncertaintyScore = scores[idx];
                var diversityScore = selected.Count == 0
                    ? _numOps.One
                    : ComputeMinDistanceToSelected(pool, idx, selected, featureSize);

                // Combined score = uncertainty * diversity
                var combinedScore = _numOps.Multiply(uncertaintyScore, diversityScore);

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
