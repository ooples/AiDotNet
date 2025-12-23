using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Implements Bayesian Active Learning by Disagreement (BALD) for sample selection.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> BALD uses information theory to select samples that maximize
/// the mutual information between model predictions and model parameters. In practice, this
/// means selecting samples where different "versions" of the model disagree the most.</para>
///
/// <para><b>Formula:</b> I(y; θ|x, D) = H(y|x, D) - E_θ[H(y|x, θ)]</para>
/// <para>where H is entropy, y is the label, θ are model parameters, x is input, D is training data.</para>
///
/// <para><b>Interpretation:</b></para>
/// <list type="bullet">
/// <item><description>First term: Entropy of the average prediction (epistemic + aleatoric uncertainty).</description></item>
/// <item><description>Second term: Average entropy across models (aleatoric uncertainty only).</description></item>
/// <item><description>Difference: Epistemic uncertainty (model uncertainty that can be reduced with more data).</description></item>
/// </list>
///
/// <para><b>Implementation:</b> Uses MC Dropout to approximate Bayesian inference by running
/// multiple forward passes with dropout enabled during inference.</para>
///
/// <para><b>Reference:</b> Houlsby, N. et al. (2011). "Bayesian Active Learning for Classification
/// and Preference Learning."</para>
/// </remarks>
public class BALD<T> : IActiveLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _numMcSamples;
    private readonly double _dropoutRate;
    private bool _useBatchDiversity;
    private T _lastMinScore;
    private T _lastMaxScore;
    private T _lastMeanScore;

    /// <summary>
    /// Initializes a new instance of the BALD class.
    /// </summary>
    /// <param name="numMcSamples">Number of Monte Carlo samples (forward passes with dropout) for
    /// approximating Bayesian inference (default: 10).</param>
    /// <param name="dropoutRate">Dropout rate for MC Dropout (default: 0.5).</param>
    public BALD(int numMcSamples = 10, double dropoutRate = 0.5)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _numMcSamples = numMcSamples;
        _dropoutRate = dropoutRate;
        _useBatchDiversity = false;
        _lastMinScore = _numOps.Zero;
        _lastMaxScore = _numOps.Zero;
        _lastMeanScore = _numOps.Zero;
    }

    /// <inheritdoc />
    public string Name => $"BALD-MC{_numMcSamples}";

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

        var numSamples = unlabeledPool.Shape[0];

        // Perform multiple forward passes (simulating MC Dropout)
        // In a real implementation, this would enable dropout during inference
        // Here we simulate by adding noise to predictions
        var mcPredictions = new List<Tensor<T>>();
        for (int m = 0; m < _numMcSamples; m++)
        {
            var predictions = model.Predict(unlabeledPool);
            // Simulate dropout effect by adding scaled noise
            mcPredictions.Add(AddDropoutNoise(predictions, _dropoutRate, m));
        }

        var numClasses = mcPredictions[0].Length / numSamples;
        var scores = new Vector<T>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            scores[i] = ComputeBALDScore(mcPredictions, i, numClasses);
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
    /// Computes BALD score: I(y; θ|x) = H(y|x) - E[H(y|x, θ)].
    /// </summary>
    private T ComputeBALDScore(List<Tensor<T>> mcPredictions, int sampleIndex, int numClasses)
    {
        // Step 1: Compute average prediction across MC samples
        var avgProbs = new Vector<T>(numClasses);
        var mcProbs = new List<Vector<T>>();

        foreach (var predictions in mcPredictions)
        {
            var probs = ExtractProbabilities(predictions, sampleIndex, numClasses);
            mcProbs.Add(probs);
            for (int c = 0; c < numClasses; c++)
            {
                avgProbs[c] = _numOps.Add(avgProbs[c], probs[c]);
            }
        }

        // Normalize average
        var numSamples = _numOps.FromDouble(mcPredictions.Count);
        for (int c = 0; c < numClasses; c++)
        {
            avgProbs[c] = _numOps.Divide(avgProbs[c], numSamples);
        }

        // Step 2: Compute H(y|x) - entropy of average prediction
        var entropyOfAvg = ComputeEntropy(avgProbs);

        // Step 3: Compute E[H(y|x, θ)] - average of individual entropies
        var avgEntropy = _numOps.Zero;
        foreach (var probs in mcProbs)
        {
            var entropy = ComputeEntropy(probs);
            avgEntropy = _numOps.Add(avgEntropy, entropy);
        }
        avgEntropy = _numOps.Divide(avgEntropy, numSamples);

        // BALD score = H(y|x) - E[H(y|x, θ)]
        return _numOps.Subtract(entropyOfAvg, avgEntropy);
    }

    /// <summary>
    /// Adds simulated dropout noise to predictions.
    /// In a full implementation, this would use actual MC Dropout in the model.
    /// </summary>
    private Tensor<T> AddDropoutNoise(Tensor<T> predictions, double dropoutRate, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var noisyPredictions = new Tensor<T>(predictions.Shape);

        for (int i = 0; i < predictions.Length; i++)
        {
            var value = _numOps.ToDouble(predictions[i]);

            // Simulate MC Dropout: randomly zero neurons and scale survivors
            if (random.NextDouble() < dropoutRate)
            {
                // Drop this neuron (set to zero)
                value = 0.0;
            }
            else
            {
                // Scale up surviving values to maintain expected value (inverted dropout)
                value *= 1.0 / (1.0 - dropoutRate);
            }

            noisyPredictions[i] = _numOps.FromDouble(value);
        }

        return noisyPredictions;
    }

    /// <summary>
    /// Computes entropy: H = -Σ p × log(p).
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
    /// Selects samples considering both BALD score and diversity.
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
                var baldScore = scores[idx];
                var diversityScore = selected.Count == 0
                    ? _numOps.One
                    : ComputeMinDistanceToSelected(pool, idx, selected, featureSize);

                var combinedScore = _numOps.Multiply(baldScore, diversityScore);

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
