using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Implements BatchBALD for joint batch selection in active learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> BatchBALD extends BALD to select batches of samples that are
/// jointly informative. Unlike naive greedy selection that picks individual high-BALD samples,
/// BatchBALD considers the joint mutual information to avoid redundant selections.</para>
///
/// <para><b>Problem with naive BALD:</b> Selecting top-k samples by individual BALD scores
/// may result in redundant samples that provide similar information.</para>
///
/// <para><b>Solution:</b> BatchBALD computes joint mutual information for candidate batches:
/// I(y₁, y₂, ..., yₖ; θ|x₁, x₂, ..., xₖ, D)</para>
///
/// <para><b>Algorithm (Greedy Approximation):</b></para>
/// <list type="number">
/// <item><description>Start with empty batch.</description></item>
/// <item><description>For each candidate, compute gain in joint mutual information if added.</description></item>
/// <item><description>Add candidate with highest marginal gain to batch.</description></item>
/// <item><description>Repeat until batch is full.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Kirsch, A., van Amersfoort, J., &amp; Gal, Y. (2019). "BatchBALD:
/// Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning."</para>
/// </remarks>
public class BatchBALD<T> : IActiveLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _numMcSamples;
    private readonly double _dropoutRate;
    private readonly int _candidatePoolSize;
    private bool _useBatchDiversity;
    private T _lastMinScore;
    private T _lastMaxScore;
    private T _lastMeanScore;

    /// <summary>
    /// Initializes a new instance of the BatchBALD class.
    /// </summary>
    /// <param name="numMcSamples">Number of MC samples for Bayesian approximation (default: 10).</param>
    /// <param name="dropoutRate">Dropout rate for MC Dropout (default: 0.5).</param>
    /// <param name="candidatePoolSize">Size of candidate pool for greedy selection (default: 100).
    /// Larger values are more accurate but slower.</param>
    public BatchBALD(int numMcSamples = 10, double dropoutRate = 0.5, int candidatePoolSize = 100)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _numMcSamples = numMcSamples;
        _dropoutRate = dropoutRate;
        _candidatePoolSize = candidatePoolSize;
        _useBatchDiversity = true; // BatchBALD inherently considers batch diversity
        _lastMinScore = _numOps.Zero;
        _lastMaxScore = _numOps.Zero;
        _lastMeanScore = _numOps.Zero;
    }

    /// <inheritdoc />
    public string Name => $"BatchBALD-MC{_numMcSamples}";

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

        var numSamples = unlabeledPool.Shape[0];
        batchSize = Math.Min(batchSize, numSamples);

        // Get MC predictions
        var mcPredictions = GetMCPredictions(model, unlabeledPool);
        var numClasses = mcPredictions[0].Length / numSamples;

        // Precompute individual BALD scores for candidate selection
        var baldScores = new Vector<T>(numSamples);
        for (int i = 0; i < numSamples; i++)
        {
            baldScores[i] = ComputeIndividualBALDScore(mcPredictions, i, numClasses);
        }

        // Select top candidates for batch selection
        var candidateIndices = GetTopCandidates(baldScores, Math.Min(_candidatePoolSize, numSamples));

        // Greedy batch selection using joint mutual information
        var selectedBatch = GreedyBatchSelection(mcPredictions, candidateIndices, batchSize, numClasses);

        UpdateStatistics(baldScores);
        return selectedBatch;
    }

    /// <inheritdoc />
    public Vector<T> ComputeInformativenessScores(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool)
    {
        _ = model ?? throw new ArgumentNullException(nameof(model));
        _ = unlabeledPool ?? throw new ArgumentNullException(nameof(unlabeledPool));

        var numSamples = unlabeledPool.Shape[0];
        var mcPredictions = GetMCPredictions(model, unlabeledPool);
        var numClasses = mcPredictions[0].Length / numSamples;
        var scores = new Vector<T>(numSamples);

        // Return individual BALD scores for informativeness
        for (int i = 0; i < numSamples; i++)
        {
            scores[i] = ComputeIndividualBALDScore(mcPredictions, i, numClasses);
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
    /// Gets MC predictions by running multiple forward passes.
    /// </summary>
    private List<Tensor<T>> GetMCPredictions(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> pool)
    {
        var mcPredictions = new List<Tensor<T>>();
        for (int m = 0; m < _numMcSamples; m++)
        {
            var predictions = model.Predict(pool);
            mcPredictions.Add(AddDropoutNoise(predictions, _dropoutRate, m));
        }
        return mcPredictions;
    }

    /// <summary>
    /// Performs greedy batch selection based on joint mutual information.
    /// </summary>
    private int[] GreedyBatchSelection(List<Tensor<T>> mcPredictions, int[] candidates, int batchSize, int numClasses)
    {
        var selected = new List<int>();
        var remaining = new HashSet<int>(candidates);

        // Cache MC probabilities for candidates
        var candidateMcProbs = new Dictionary<int, List<Vector<T>>>();
        foreach (var idx in candidates)
        {
            candidateMcProbs[idx] = new List<Vector<T>>();
            foreach (var predictions in mcPredictions)
            {
                candidateMcProbs[idx].Add(ExtractProbabilities(predictions, idx, numClasses));
            }
        }

        while (selected.Count < batchSize && remaining.Count > 0)
        {
            var bestCandidate = -1;
            var bestGain = _numOps.MinValue;

            foreach (var candidate in remaining)
            {
                // Compute marginal gain of adding this candidate
                var gain = ComputeMarginalGain(candidateMcProbs, selected, candidate, numClasses);

                if (_numOps.GreaterThan(gain, bestGain))
                {
                    bestGain = gain;
                    bestCandidate = candidate;
                }
            }

            if (bestCandidate >= 0)
            {
                selected.Add(bestCandidate);
                remaining.Remove(bestCandidate);
            }
            else
            {
                break;
            }
        }

        return [.. selected];
    }

    /// <summary>
    /// Computes the marginal gain in joint mutual information when adding a candidate.
    /// </summary>
    private T ComputeMarginalGain(
        Dictionary<int, List<Vector<T>>> candidateMcProbs,
        List<int> currentBatch,
        int candidate,
        int numClasses)
    {
        if (currentBatch.Count == 0)
        {
            // First sample: gain is the individual BALD score
            return ComputeIndividualBALDFromProbs(candidateMcProbs[candidate], numClasses);
        }

        // Compute joint MI with candidate vs without
        // This is a simplified approximation - full BatchBALD uses more complex computation
        var batchProbs = new List<List<Vector<T>>>();
        foreach (var idx in currentBatch)
        {
            batchProbs.Add(candidateMcProbs[idx]);
        }

        var jointMI = ComputeJointMI(batchProbs, numClasses);
        batchProbs.Add(candidateMcProbs[candidate]);
        var jointMIWithCandidate = ComputeJointMI(batchProbs, numClasses);

        return _numOps.Subtract(jointMIWithCandidate, jointMI);
    }

    /// <summary>
    /// Computes approximate joint mutual information for a batch.
    /// </summary>
    private T ComputeJointMI(List<List<Vector<T>>> batchMcProbs, int numClasses)
    {
        if (batchMcProbs.Count == 0)
            return _numOps.Zero;

        // Simplified approximation: sum of individual MI minus redundancy
        var totalMI = _numOps.Zero;
        var redundancy = _numOps.Zero;

        for (int i = 0; i < batchMcProbs.Count; i++)
        {
            var mi = ComputeIndividualBALDFromProbs(batchMcProbs[i], numClasses);
            totalMI = _numOps.Add(totalMI, mi);

            // Compute redundancy with previously selected samples
            for (int j = 0; j < i; j++)
            {
                var pairRedundancy = ComputePairwiseRedundancy(batchMcProbs[i], batchMcProbs[j], numClasses);
                redundancy = _numOps.Add(redundancy, pairRedundancy);
            }
        }

        // Joint MI ≈ Sum of individual MI - redundancy
        return _numOps.Subtract(totalMI, redundancy);
    }

    /// <summary>
    /// Computes pairwise redundancy between two samples based on prediction agreement.
    /// </summary>
    private T ComputePairwiseRedundancy(List<Vector<T>> probs1, List<Vector<T>> probs2, int numClasses)
    {
        // Compute average prediction agreement
        var agreement = 0.0;

        for (int m = 0; m < probs1.Count; m++)
        {
            var dotProduct = 0.0;
            for (int c = 0; c < numClasses; c++)
            {
                dotProduct += _numOps.ToDouble(probs1[m][c]) * _numOps.ToDouble(probs2[m][c]);
            }
            agreement += dotProduct;
        }

        agreement /= probs1.Count;

        // Higher agreement = more redundancy
        // Scale to be a small fraction of MI
        return _numOps.FromDouble(agreement * 0.1);
    }

    /// <summary>
    /// Computes individual BALD score from precomputed MC probabilities.
    /// </summary>
    private T ComputeIndividualBALDFromProbs(List<Vector<T>> mcProbs, int numClasses)
    {
        // Average prediction
        var avgProbs = new Vector<T>(numClasses);
        foreach (var probs in mcProbs)
        {
            for (int c = 0; c < numClasses; c++)
            {
                avgProbs[c] = _numOps.Add(avgProbs[c], probs[c]);
            }
        }
        var numSamples = _numOps.FromDouble(mcProbs.Count);
        for (int c = 0; c < numClasses; c++)
        {
            avgProbs[c] = _numOps.Divide(avgProbs[c], numSamples);
        }

        // Entropy of average
        var entropyOfAvg = ComputeEntropy(avgProbs);

        // Average of entropies
        var avgEntropy = _numOps.Zero;
        foreach (var probs in mcProbs)
        {
            avgEntropy = _numOps.Add(avgEntropy, ComputeEntropy(probs));
        }
        avgEntropy = _numOps.Divide(avgEntropy, numSamples);

        return _numOps.Subtract(entropyOfAvg, avgEntropy);
    }

    /// <summary>
    /// Computes individual BALD score from MC predictions.
    /// </summary>
    private T ComputeIndividualBALDScore(List<Tensor<T>> mcPredictions, int sampleIndex, int numClasses)
    {
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

        var numSamples = _numOps.FromDouble(mcPredictions.Count);
        for (int c = 0; c < numClasses; c++)
        {
            avgProbs[c] = _numOps.Divide(avgProbs[c], numSamples);
        }

        var entropyOfAvg = ComputeEntropy(avgProbs);

        var avgEntropy = _numOps.Zero;
        foreach (var probs in mcProbs)
        {
            avgEntropy = _numOps.Add(avgEntropy, ComputeEntropy(probs));
        }
        avgEntropy = _numOps.Divide(avgEntropy, numSamples);

        return _numOps.Subtract(entropyOfAvg, avgEntropy);
    }

    /// <summary>
    /// Gets top candidates based on individual BALD scores.
    /// </summary>
    private int[] GetTopCandidates(Vector<T> scores, int topK)
    {
        var indexedScores = new List<(int Index, T Score)>();
        for (int i = 0; i < scores.Length; i++)
        {
            indexedScores.Add((i, scores[i]));
        }

        return indexedScores
            .OrderByDescending(x => _numOps.ToDouble(x.Score))
            .Take(topK)
            .Select(x => x.Index)
            .ToArray();
    }

    /// <summary>
    /// Adds simulated dropout noise to predictions.
    /// </summary>
    private Tensor<T> AddDropoutNoise(Tensor<T> predictions, double dropoutRate, int seed)
    {
        var random = new Random(seed);
        var noisyPredictions = new Tensor<T>(predictions.Shape);

        for (int i = 0; i < predictions.Length; i++)
        {
            var value = _numOps.ToDouble(predictions[i]);

            if (random.NextDouble() < dropoutRate)
            {
                value *= 1.0 / (1.0 - dropoutRate);
            }
            else
            {
                value += (random.NextDouble() - 0.5) * 0.1;
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
