using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Optimization)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Active Learning Literature Survey", "https://minds.wisconsin.edu/handle/1793/60660", Year = 2009, Authors = "Burr Settles")]
[ComponentType(ComponentType.ActiveLearner)]
[PipelineStage(PipelineStage.Training)]
public class RandomSampling<T> : IActiveLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Random _random;

    /// <summary>
    /// Mixed into every sample's score hash, so the seed actually selects the random draw.
    /// </summary>
    /// <remarks>
    /// Scores are hashed from the pool data so one instance always ranks a pool the same way (the
    /// k=1 pick is always inside the k=5 picks). Without a salt that hash ignored the seed entirely:
    /// every RandomSampling, whatever its seed, selected the identical samples, so "random" sampling
    /// was a fixed ordering of the data and a caller's seed had no effect.
    /// </remarks>
    private readonly uint _salt;
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
        _salt = unchecked((uint)_random.Next() ^ ((uint)_random.Next() << 16));
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

        // Compute informativeness scores and select top-k indices.
        // For RandomSampling, scores are computed from a seeded hash of the pool data,
        // ensuring consistency: SelectSamples(k=1) result appears in SelectSamples(k=5).
        var scores = ComputeInformativenessScores(model, unlabeledPool);

        // Select top-scoring indices
        var indexedScores = new (int index, T score)[numSamples];
        for (int i = 0; i < numSamples; i++)
            indexedScores[i] = (i, scores[i]);

        Array.Sort(indexedScores, (a, b) =>
        {
            if (_numOps.GreaterThan(a.score, b.score)) return -1;
            if (_numOps.GreaterThan(b.score, a.score)) return 1;
            return 0;
        });

        var result = new int[batchSize];
        for (int i = 0; i < batchSize; i++)
            result[i] = indexedScores[i].index;

        return result;
    }

    /// <inheritdoc />
    public Vector<T> ComputeInformativenessScores(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool)
    {
        Guard.NotNull(model);
        Guard.NotNull(unlabeledPool);

        var numSamples = unlabeledPool.Shape[0];
        var scores = new Vector<T>(numSamples);

        // Random sampling assigns pseudo-random scores derived from the pool data itself.
        // This ensures consistency: the same pool always produces the same scores,
        // so SelectSamples(k=1) result appears in SelectSamples(k=5) result.
        // The randomness comes from hashing each sample's data values.
        for (int i = 0; i < numSamples; i++)
        {
            // Hash the sample's features, salted by this instance's seed, to produce a deterministic
            // pseudo-random score. The salt is what makes two seeds rank the pool differently.
            uint hash = unchecked((uint)(i * 2654435761L) ^ _salt); // Knuth multiplicative hash seed
            hash = unchecked((uint)(hash * 2654435761L));
            int featureDim = unlabeledPool.Length / numSamples;
            for (int f = 0; f < Math.Min(featureDim, 8); f++)
            {
                int flatIdx = i * featureDim + f;
                if (flatIdx < unlabeledPool.Length)
                {
                    long bits = BitConverter.DoubleToInt64Bits(_numOps.ToDouble(unlabeledPool[flatIdx]));
                    hash ^= (uint)(bits >> 32) ^ (uint)bits;
                    hash = (uint)(hash * 2654435761L);
                }
            }
            // Map hash to [0, 1] range
            scores[i] = _numOps.FromDouble((uint)hash / (double)uint.MaxValue);
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
