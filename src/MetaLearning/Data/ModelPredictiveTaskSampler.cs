using AiDotNet.Data.Structures;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Data;

/// <summary>
/// Model Predictive Task Sampling (MPTS): predicts which tasks will yield the greatest
/// learning signal by maintaining a posterior estimate of per-task adaptation risk,
/// then sampling tasks that balance exploration and exploitation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Instead of sampling tasks randomly, MPTS tries to predict
/// which tasks will help the model learn the most. It keeps track of how much the model
/// improved on each task and favors tasks where improvement is likely highest.</para>
/// <para><b>Reference:</b> Model Predictive Task Sampling (2025).</para>
/// </remarks>
public class ModelPredictiveTaskSampler<T, TInput, TOutput> : ITaskSampler<T, TInput, TOutput>
{
    private readonly IMetaDataset<T, TInput, TOutput> _dataset;
    private readonly double _explorationWeight;
    private readonly double _ucbScale;
    private readonly Dictionary<int, double> _meanRewards;
    private readonly Dictionary<int, int> _pullCounts;
    private int _totalPulls;
    private Random _rng;

    /// <inheritdoc/>
    public int NumWays { get; }

    /// <inheritdoc/>
    public int NumShots { get; }

    /// <inheritdoc/>
    public int NumQueryPerClass { get; }

    /// <summary>
    /// Creates a model-predictive task sampler.
    /// </summary>
    /// <param name="dataset">The meta-dataset to sample from.</param>
    /// <param name="numWays">Number of classes per task.</param>
    /// <param name="numShots">Number of support examples per class.</param>
    /// <param name="numQueryPerClass">Number of query examples per class.</param>
    /// <param name="explorationWeight">Weight for exploration bonus in UCB. Higher values
    /// increase exploration of rarely-sampled tasks. Default: 1.0.</param>
    /// <param name="ucbScale">Scale factor for the UCB confidence bound. Default: 2.0.</param>
    /// <param name="seed">Optional random seed.</param>
    public ModelPredictiveTaskSampler(
        IMetaDataset<T, TInput, TOutput> dataset,
        int numWays = 5,
        int numShots = 1,
        int numQueryPerClass = 15,
        double explorationWeight = 1.0,
        double ucbScale = 2.0,
        int? seed = null)
    {
        _dataset = dataset;
        NumWays = numWays;
        NumShots = numShots;
        NumQueryPerClass = numQueryPerClass;
        _explorationWeight = explorationWeight;
        _ucbScale = ucbScale;
        _meanRewards = new Dictionary<int, double>();
        _pullCounts = new Dictionary<int, int>();
        _rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc/>
    public TaskBatch<T, TInput, TOutput> SampleBatch(int batchSize)
    {
        var tasks = new IMetaLearningTask<T, TInput, TOutput>[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            tasks[i] = SampleOne().Task;
        }
        return new TaskBatch<T, TInput, TOutput>(tasks, BatchingStrategy.Adaptive);
    }

    /// <summary>Number of candidate episodes to sample for UCB selection.</summary>
    private const int UcbCandidates = 5;

    /// <summary>Fraction of calls that use pure random exploration.</summary>
    private const double ExplorationRate = 0.1;

    /// <inheritdoc/>
    public IEpisode<T, TInput, TOutput> SampleOne()
    {
        // Pure exploration: random sample
        if (_meanRewards.Count == 0 || _rng.NextDouble() < ExplorationRate)
        {
            _totalPulls++;
            return _dataset.SampleEpisode(NumWays, NumShots, NumQueryPerClass);
        }

        // Exploit: sample multiple candidates and pick the one with highest UCB score
        IEpisode<T, TInput, TOutput>? bestEpisode = null;
        double bestUcb = double.NegativeInfinity;

        for (int c = 0; c < UcbCandidates; c++)
        {
            var candidate = _dataset.SampleEpisode(NumWays, NumShots, NumQueryPerClass);
            // Use a stable task signature (hash of Domain + EpisodeId modulo a fixed arm space)
            // to prevent unbounded growth while allowing meaningful exploitation
            int armKey = ComputeArmKey(candidate);

            double ucb;
            if (_pullCounts.TryGetValue(armKey, out int pulls) && pulls > 0)
            {
                double exploitation = _meanRewards[armKey];
                double exploration = _explorationWeight * Math.Sqrt(
                    _ucbScale * Math.Log(_totalPulls + 1) / Math.Max(1, pulls));
                ucb = exploitation + exploration;
            }
            else
            {
                // Never-seen arms get maximum exploration bonus
                ucb = double.MaxValue;
            }

            if (ucb > bestUcb)
            {
                bestUcb = ucb;
                bestEpisode = candidate;
            }
        }

        bestEpisode!.Difficulty = bestUcb == double.MaxValue ? 1.0 : Math.Max(0, Math.Min(1, bestUcb));
        _totalPulls++;
        return bestEpisode;
    }

    /// <inheritdoc/>
    public void UpdateWithFeedback(IReadOnlyList<IEpisode<T, TInput, TOutput>> episodes, IReadOnlyList<double> losses)
    {
        for (int i = 0; i < Math.Min(episodes.Count, losses.Count); i++)
        {
            episodes[i].LastLoss = losses[i];
            episodes[i].SampleCount++;

            // Reward = loss reduction potential (higher loss = more potential improvement)
            double reward = losses[i];

            // Use stable arm key instead of unbounded EpisodeId
            int armKey = ComputeArmKey(episodes[i]);

            if (!_pullCounts.ContainsKey(armKey))
            {
                _pullCounts[armKey] = 0;
                _meanRewards[armKey] = 0;
            }

            // Incremental mean update
            _pullCounts[armKey]++;
            _meanRewards[armKey] += (reward - _meanRewards[armKey]) / _pullCounts[armKey];
        }
    }

    /// <summary>
    /// Computes a stable arm key for bandit statistics from an episode's domain and structure.
    /// Uses a bounded hash to prevent unbounded memory growth.
    /// </summary>
    private static int ComputeArmKey(IEpisode<T, TInput, TOutput> episode)
    {
        // Combine domain (if available) with a hash of the task structure
        // This groups episodes from similar tasks together for meaningful exploitation
        int hash = 17;
        if (episode.Domain is not null)
            hash = hash * 31 + episode.Domain.GetHashCode(StringComparison.Ordinal);
        // Use episode metadata or difficulty as a secondary discriminator
        if (episode.Difficulty.HasValue)
            hash = hash * 31 + episode.Difficulty.Value.GetHashCode();
        // Bound the arm space to prevent unbounded growth (1024 arms)
        return Math.Abs(hash) % 1024;
    }

    /// <inheritdoc/>
    public void SetSeed(int seed)
    {
        _rng = RandomHelper.CreateSeededRandom(seed);
        _dataset.SetSeed(seed);
    }
}
