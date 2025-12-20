using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Implements Hybrid Sampling that combines multiple active learning strategies.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Hybrid sampling combines the benefits of multiple active learning
/// strategies. For example, uncertainty sampling might select many similar samples near the
/// decision boundary, while diversity sampling ensures good coverage. Combining them gets the
/// best of both worlds.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>Compute scores from multiple strategies (e.g., uncertainty and diversity).</description></item>
/// <item><description>Combine scores using a specified method (weighted sum, product, rank fusion).</description></item>
/// <item><description>Select samples based on the combined scores.</description></item>
/// </list>
///
/// <para><b>Common combinations:</b></para>
/// <list type="bullet">
/// <item><description><b>Uncertainty + Diversity:</b> Select uncertain samples that are also diverse.
/// This is the most common combination.</description></item>
/// <item><description><b>Uncertainty + Expected Change:</b> Select samples that are both uncertain
/// and would significantly change the model.</description></item>
/// <item><description><b>Multi-strategy ensemble:</b> Use rank fusion to combine any number of strategies.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Settles, "Active Learning Literature Survey" (2009).</para>
/// </remarks>
public class HybridSampling<T> : IActiveLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly List<(IActiveLearningStrategy<T> Strategy, double Weight)> _strategies;
    private readonly CombinationMethod _method;
    private bool _useBatchDiversity;
    private T _lastMinScore;
    private T _lastMaxScore;
    private T _lastMeanScore;

    /// <summary>
    /// Defines methods for combining strategy scores.
    /// </summary>
    public enum CombinationMethod
    {
        /// <summary>Weighted sum of normalized scores.</summary>
        WeightedSum,
        /// <summary>Product of scores (samples must be good in all strategies).</summary>
        Product,
        /// <summary>Rank-based fusion (combines rankings, not scores).</summary>
        RankFusion,
        /// <summary>Maximum score across strategies.</summary>
        Maximum,
        /// <summary>Minimum score across strategies (most conservative).</summary>
        Minimum
    }

    /// <summary>
    /// Initializes a new instance of the HybridSampling class.
    /// </summary>
    /// <param name="strategies">List of strategies with their weights.</param>
    /// <param name="method">Method for combining scores (default: WeightedSum).</param>
    public HybridSampling(
        IEnumerable<(IActiveLearningStrategy<T> Strategy, double Weight)> strategies,
        CombinationMethod method = CombinationMethod.WeightedSum)
    {
        _ = strategies ?? throw new ArgumentNullException(nameof(strategies));

        _numOps = MathHelper.GetNumericOperations<T>();
        _strategies = [.. strategies];
        _method = method;
        _useBatchDiversity = false;
        _lastMinScore = _numOps.Zero;
        _lastMaxScore = _numOps.Zero;
        _lastMeanScore = _numOps.Zero;

        if (_strategies.Count == 0)
        {
            throw new ArgumentException("At least one strategy must be provided.", nameof(strategies));
        }
    }

    /// <summary>
    /// Creates a default hybrid strategy combining uncertainty sampling and diversity sampling.
    /// </summary>
    /// <param name="uncertaintyWeight">Weight for uncertainty (default: 0.7).</param>
    /// <param name="diversityWeight">Weight for diversity (default: 0.3).</param>
    /// <returns>A configured HybridSampling instance.</returns>
    public static HybridSampling<T> CreateUncertaintyDiversity(
        double uncertaintyWeight = 0.7,
        double diversityWeight = 0.3)
    {
        var strategies = new List<(IActiveLearningStrategy<T>, double)>
        {
            (new UncertaintySampling<T>(UncertaintySampling<T>.UncertaintyMeasure.Entropy), uncertaintyWeight),
            (new DiversitySampling<T>(DiversitySampling<T>.DiversityMethod.KCenterGreedy), diversityWeight)
        };

        return new HybridSampling<T>(strategies, CombinationMethod.WeightedSum);
    }

    /// <inheritdoc />
    public string Name
    {
        get
        {
            var names = string.Join("+", _strategies.Select(s => s.Strategy.Name));
            return $"Hybrid[{names}]-{_method}";
        }
    }

    /// <inheritdoc />
    public bool UseBatchDiversity
    {
        get => _useBatchDiversity;
        set => _useBatchDiversity = value;
    }

    /// <summary>
    /// Gets the strategies used in this hybrid sampler.
    /// </summary>
    public IReadOnlyList<(IActiveLearningStrategy<T> Strategy, double Weight)> Strategies => _strategies;

    /// <inheritdoc />
    public int[] SelectSamples(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool, int batchSize)
    {
        _ = model ?? throw new ArgumentNullException(nameof(model));
        _ = unlabeledPool ?? throw new ArgumentNullException(nameof(unlabeledPool));

        var scores = ComputeInformativenessScores(model, unlabeledPool);
        var numSamples = unlabeledPool.Shape[0];
        batchSize = Math.Min(batchSize, numSamples);

        return SelectTopScoring(scores, batchSize);
    }

    /// <inheritdoc />
    public Vector<T> ComputeInformativenessScores(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool)
    {
        _ = model ?? throw new ArgumentNullException(nameof(model));
        _ = unlabeledPool ?? throw new ArgumentNullException(nameof(unlabeledPool));

        var numSamples = unlabeledPool.Shape[0];

        // Collect scores from all strategies
        var allScores = new List<Vector<T>>();
        foreach (var (strategy, _) in _strategies)
        {
            var scores = strategy.ComputeInformativenessScores(model, unlabeledPool);
            allScores.Add(scores);
        }

        // Combine scores using the specified method
        var combinedScores = _method switch
        {
            CombinationMethod.WeightedSum => CombineWeightedSum(allScores, numSamples),
            CombinationMethod.Product => CombineProduct(allScores, numSamples),
            CombinationMethod.RankFusion => CombineRankFusion(allScores, numSamples),
            CombinationMethod.Maximum => CombineMaximum(allScores, numSamples),
            CombinationMethod.Minimum => CombineMinimum(allScores, numSamples),
            _ => CombineWeightedSum(allScores, numSamples)
        };

        UpdateStatistics(combinedScores);
        return combinedScores;
    }

    /// <inheritdoc />
    public Dictionary<string, T> GetSelectionStatistics()
    {
        var stats = new Dictionary<string, T>
        {
            ["MinScore"] = _lastMinScore,
            ["MaxScore"] = _lastMaxScore,
            ["MeanScore"] = _lastMeanScore,
            ["NumStrategies"] = _numOps.FromDouble(_strategies.Count)
        };

        // Include individual strategy stats
        for (int i = 0; i < _strategies.Count; i++)
        {
            var stratStats = _strategies[i].Strategy.GetSelectionStatistics();
            foreach (var (key, value) in stratStats)
            {
                stats[$"Strategy{i}_{key}"] = value;
            }
        }

        return stats;
    }

    /// <summary>
    /// Combines scores using weighted sum of normalized scores.
    /// </summary>
    private Vector<T> CombineWeightedSum(List<Vector<T>> allScores, int numSamples)
    {
        var result = new Vector<T>(numSamples);

        for (int s = 0; s < _strategies.Count; s++)
        {
            var scores = NormalizeScores(allScores[s]);
            var weight = _numOps.FromDouble(_strategies[s].Weight);

            for (int i = 0; i < numSamples; i++)
            {
                var weighted = _numOps.Multiply(scores[i], weight);
                result[i] = _numOps.Add(result[i], weighted);
            }
        }

        return result;
    }

    /// <summary>
    /// Combines scores using product (geometric mean-like).
    /// </summary>
    private Vector<T> CombineProduct(List<Vector<T>> allScores, int numSamples)
    {
        var result = new Vector<T>(numSamples);
        var epsilon = _numOps.FromDouble(1e-10);

        // Initialize with ones
        for (int i = 0; i < numSamples; i++)
        {
            result[i] = _numOps.One;
        }

        foreach (var scores in allScores)
        {
            var normalized = NormalizeScores(scores);
            for (int i = 0; i < numSamples; i++)
            {
                // Add epsilon to avoid zero products
                var value = _numOps.Add(normalized[i], epsilon);
                result[i] = _numOps.Multiply(result[i], value);
            }
        }

        return result;
    }

    /// <summary>
    /// Combines scores using rank fusion (Borda count).
    /// </summary>
    private Vector<T> CombineRankFusion(List<Vector<T>> allScores, int numSamples)
    {
        var result = new Vector<T>(numSamples);

        for (int s = 0; s < _strategies.Count; s++)
        {
            var scores = allScores[s];
            var weight = _strategies[s].Weight;

            // Compute ranks
            var indexedScores = new List<(int Index, T Score)>();
            for (int i = 0; i < numSamples; i++)
            {
                indexedScores.Add((i, scores[i]));
            }

            var ranked = indexedScores
                .OrderByDescending(x => _numOps.ToDouble(x.Score))
                .Select((x, rank) => (x.Index, Rank: numSamples - rank))
                .ToDictionary(x => x.Index, x => x.Rank);

            // Add weighted ranks
            for (int i = 0; i < numSamples; i++)
            {
                var rankScore = _numOps.FromDouble(ranked[i] * weight);
                result[i] = _numOps.Add(result[i], rankScore);
            }
        }

        return result;
    }

    /// <summary>
    /// Combines scores using maximum (optimistic combination).
    /// </summary>
    private Vector<T> CombineMaximum(List<Vector<T>> allScores, int numSamples)
    {
        var result = new Vector<T>(numSamples);

        // Normalize all scores first
        var normalizedScores = allScores.Select(NormalizeScores).ToList();

        for (int i = 0; i < numSamples; i++)
        {
            var maxScore = _numOps.MinValue;
            foreach (var scores in normalizedScores)
            {
                if (_numOps.GreaterThan(scores[i], maxScore))
                {
                    maxScore = scores[i];
                }
            }
            result[i] = maxScore;
        }

        return result;
    }

    /// <summary>
    /// Combines scores using minimum (conservative combination).
    /// </summary>
    private Vector<T> CombineMinimum(List<Vector<T>> allScores, int numSamples)
    {
        var result = new Vector<T>(numSamples);

        // Normalize all scores first
        var normalizedScores = allScores.Select(NormalizeScores).ToList();

        for (int i = 0; i < numSamples; i++)
        {
            var minScore = _numOps.MaxValue;
            foreach (var scores in normalizedScores)
            {
                if (_numOps.LessThan(scores[i], minScore))
                {
                    minScore = scores[i];
                }
            }
            result[i] = minScore;
        }

        return result;
    }

    /// <summary>
    /// Normalizes scores to [0, 1] range using min-max normalization.
    /// </summary>
    private Vector<T> NormalizeScores(Vector<T> scores)
    {
        if (scores.Length == 0)
        {
            return scores;
        }

        var min = scores[0];
        var max = scores[0];

        for (int i = 1; i < scores.Length; i++)
        {
            if (_numOps.LessThan(scores[i], min)) min = scores[i];
            if (_numOps.GreaterThan(scores[i], max)) max = scores[i];
        }

        var range = _numOps.Subtract(max, min);
        var epsilon = _numOps.FromDouble(1e-10);

        // If all scores are the same, return 0.5 for all
        if (_numOps.LessThan(range, epsilon))
        {
            var result = new Vector<T>(scores.Length);
            var half = _numOps.FromDouble(0.5);
            for (int i = 0; i < scores.Length; i++)
            {
                result[i] = half;
            }
            return result;
        }

        var normalized = new Vector<T>(scores.Length);
        for (int i = 0; i < scores.Length; i++)
        {
            normalized[i] = _numOps.Divide(_numOps.Subtract(scores[i], min), range);
        }

        return normalized;
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
