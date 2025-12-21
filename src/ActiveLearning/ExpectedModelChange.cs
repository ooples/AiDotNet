using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Implements Expected Model Change (EMC) for active learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Expected Model Change selects samples that would cause the largest
/// change to the model's parameters if they were labeled and used for training. The intuition is
/// that samples which significantly change the model provide the most learning value.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>For each unlabeled sample, compute the expected gradient if it were labeled
/// with each possible class.</description></item>
/// <item><description>Weight each gradient by the probability of that class being correct.</description></item>
/// <item><description>The expected gradient magnitude indicates how much the model would change.</description></item>
/// <item><description>Select samples with the largest expected change.</description></item>
/// </list>
///
/// <para><b>Variants:</b></para>
/// <list type="bullet">
/// <item><description><b>EGL (Expected Gradient Length):</b> Uses the expected L2 norm of the gradient.</description></item>
/// <item><description><b>EMCM (Expected Model Change Maximization):</b> Uses gradient magnitude weighted by
/// current model confidence.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Settles et al., "Multiple-Instance Active Learning" (2008). ICML.</para>
/// </remarks>
public class ExpectedModelChange<T> : IActiveLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly ChangeMetric _metric;
    private bool _useBatchDiversity;
    private T _lastMinScore;
    private T _lastMaxScore;
    private T _lastMeanScore;

    /// <summary>
    /// Defines the change metric to use.
    /// </summary>
    public enum ChangeMetric
    {
        /// <summary>Expected L2 norm of the gradient.</summary>
        ExpectedGradientLength,
        /// <summary>Maximum gradient length across all possible labels.</summary>
        MaxGradientLength,
        /// <summary>Variance of gradient lengths across labels.</summary>
        GradientVariance
    }

    /// <summary>
    /// Initializes a new instance of the ExpectedModelChange class.
    /// </summary>
    /// <param name="metric">The change metric to use (default: ExpectedGradientLength).</param>
    public ExpectedModelChange(ChangeMetric metric = ChangeMetric.ExpectedGradientLength)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _metric = metric;
        _useBatchDiversity = false;
        _lastMinScore = _numOps.Zero;
        _lastMaxScore = _numOps.Zero;
        _lastMeanScore = _numOps.Zero;
    }

    /// <inheritdoc />
    public string Name => $"ExpectedModelChange-{_metric}";

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
            scores[i] = ComputeExpectedChange(probs, numClasses);
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
    /// Computes the expected model change for a single sample.
    /// </summary>
    private T ComputeExpectedChange(Vector<T> probabilities, int numClasses)
    {
        return _metric switch
        {
            ChangeMetric.ExpectedGradientLength => ComputeEGL(probabilities, numClasses),
            ChangeMetric.MaxGradientLength => ComputeMaxGradient(probabilities, numClasses),
            ChangeMetric.GradientVariance => ComputeGradientVariance(probabilities, numClasses),
            _ => ComputeEGL(probabilities, numClasses)
        };
    }

    /// <summary>
    /// Computes Expected Gradient Length (EGL).
    /// </summary>
    /// <remarks>
    /// For each possible label y, the gradient of cross-entropy loss is (p - one_hot(y)).
    /// The expected gradient length is: Σ_y P(y|x) * ||p - one_hot(y)||
    /// This simplifies to: Σ_y P(y|x) * sqrt(Σ_c (p_c - δ_{cy})²)
    /// </remarks>
    private T ComputeEGL(Vector<T> probabilities, int numClasses)
    {
        var egl = _numOps.Zero;

        // For each possible label
        for (int y = 0; y < numClasses; y++)
        {
            // Compute gradient magnitude if label were y
            var gradMagnitudeSq = _numOps.Zero;
            for (int c = 0; c < numClasses; c++)
            {
                // Gradient component: p_c - δ_{cy}
                var delta = (c == y) ? _numOps.One : _numOps.Zero;
                var gradComponent = _numOps.Subtract(probabilities[c], delta);
                var squared = _numOps.Multiply(gradComponent, gradComponent);
                gradMagnitudeSq = _numOps.Add(gradMagnitudeSq, squared);
            }

            // Weight by probability of this label
            var gradMagnitude = _numOps.FromDouble(Math.Sqrt(_numOps.ToDouble(gradMagnitudeSq)));
            var weighted = _numOps.Multiply(probabilities[y], gradMagnitude);
            egl = _numOps.Add(egl, weighted);
        }

        return egl;
    }

    /// <summary>
    /// Computes maximum gradient length across all possible labels.
    /// </summary>
    private T ComputeMaxGradient(Vector<T> probabilities, int numClasses)
    {
        var maxGrad = _numOps.Zero;

        for (int y = 0; y < numClasses; y++)
        {
            var gradMagnitudeSq = _numOps.Zero;
            for (int c = 0; c < numClasses; c++)
            {
                var delta = (c == y) ? _numOps.One : _numOps.Zero;
                var gradComponent = _numOps.Subtract(probabilities[c], delta);
                var squared = _numOps.Multiply(gradComponent, gradComponent);
                gradMagnitudeSq = _numOps.Add(gradMagnitudeSq, squared);
            }

            var gradMagnitude = _numOps.FromDouble(Math.Sqrt(_numOps.ToDouble(gradMagnitudeSq)));
            if (_numOps.GreaterThan(gradMagnitude, maxGrad))
            {
                maxGrad = gradMagnitude;
            }
        }

        return maxGrad;
    }

    /// <summary>
    /// Computes variance of gradient lengths across possible labels.
    /// </summary>
    private T ComputeGradientVariance(Vector<T> probabilities, int numClasses)
    {
        var gradLengths = new Vector<T>(numClasses);

        // Compute gradient length for each possible label
        for (int y = 0; y < numClasses; y++)
        {
            var gradMagnitudeSq = _numOps.Zero;
            for (int c = 0; c < numClasses; c++)
            {
                var delta = (c == y) ? _numOps.One : _numOps.Zero;
                var gradComponent = _numOps.Subtract(probabilities[c], delta);
                var squared = _numOps.Multiply(gradComponent, gradComponent);
                gradMagnitudeSq = _numOps.Add(gradMagnitudeSq, squared);
            }
            gradLengths[y] = _numOps.FromDouble(Math.Sqrt(_numOps.ToDouble(gradMagnitudeSq)));
        }

        // Compute mean
        var mean = _numOps.Zero;
        for (int y = 0; y < numClasses; y++)
        {
            mean = _numOps.Add(mean, gradLengths[y]);
        }
        mean = _numOps.Divide(mean, _numOps.FromDouble(numClasses));

        // Compute variance
        var variance = _numOps.Zero;
        for (int y = 0; y < numClasses; y++)
        {
            var diff = _numOps.Subtract(gradLengths[y], mean);
            var squared = _numOps.Multiply(diff, diff);
            variance = _numOps.Add(variance, squared);
        }

        return _numOps.Divide(variance, _numOps.FromDouble(numClasses));
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
    /// Selects samples considering both expected change and diversity.
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
                var changeScore = scores[idx];
                var diversityScore = selected.Count == 0
                    ? _numOps.One
                    : ComputeMinDistanceToSelected(pool, idx, selected, featureSize);

                var combinedScore = _numOps.Multiply(changeScore, diversityScore);

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
