using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Implements random sampling for active learning (baseline strategy).
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Random sampling is the simplest active learning strategy.
/// It randomly selects samples from the unlabeled pool without considering model predictions
/// or any informativeness measure. This serves as a baseline for comparing other strategies.</para>
///
/// <para><b>When to use:</b></para>
/// <list type="bullet">
/// <item><description>As a baseline to compare against more sophisticated strategies.</description></item>
/// <item><description>When you want to ensure unbiased sampling from the data distribution.</description></item>
/// <item><description>When model predictions are unreliable (early training stages).</description></item>
/// </list>
///
/// <para><b>Complexity:</b> O(n) for selection where n is the pool size.</para>
/// <para><b>Reference:</b> Standard baseline in active learning literature.</para>
/// </remarks>
public class RandomSampling<T> : IActiveLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Random _random;
    private bool _useBatchDiversity;
    private T _lastMinScore;
    private T _lastMaxScore;
    private T _lastMeanScore;

    /// <summary>
    /// Initializes a new instance of the RandomSampling class.
    /// </summary>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public RandomSampling(int? seed = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        _useBatchDiversity = false;
        _lastMinScore = _numOps.Zero;
        _lastMaxScore = _numOps.Zero;
        _lastMeanScore = _numOps.Zero;
    }

    /// <inheritdoc />
    public string Name => "RandomSampling";

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

        var numSamples = unlabeledPool.Shape[0];
        batchSize = Math.Min(batchSize, numSamples);

        // Generate random scores for statistics tracking
        var scores = ComputeInformativenessScores(model, unlabeledPool);

        // Generate random permutation and take first batchSize elements
        var indices = Enumerable.Range(0, numSamples).ToList();

        // Fisher-Yates shuffle
        for (int i = indices.Count - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        return indices.Take(batchSize).ToArray();
    }

    /// <inheritdoc />
    public Vector<T> ComputeInformativenessScores(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool)
    {
        Guard.NotNull(model);
        Guard.NotNull(unlabeledPool);

        var numSamples = unlabeledPool.Shape[0];
        var scores = new Vector<T>(numSamples);

        // Random sampling assigns uniform scores (all samples are equally informative)
        // The randomness comes from the shuffle in SelectSamples, not from the scores
        var uniformScore = _numOps.FromDouble(1.0 / numSamples);
        for (int i = 0; i < numSamples; i++)
        {
            scores[i] = uniformScore;
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
