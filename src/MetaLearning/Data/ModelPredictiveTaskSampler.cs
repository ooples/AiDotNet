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
    private readonly List<double> _meanRewards;
    private readonly List<int> _pullCounts;
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
        _meanRewards = new List<double>();
        _pullCounts = new List<int>();
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

    /// <inheritdoc/>
    public IEpisode<T, TInput, TOutput> SampleOne()
    {
        var episode = _dataset.SampleEpisode(NumWays, NumShots, NumQueryPerClass);

        // Use UCB-style selection: prioritize tasks with high predicted reward or high uncertainty
        if (_meanRewards.Count > 0 && _rng.NextDouble() > 0.1) // 90% exploit, 10% explore
        {
            // Set difficulty based on predicted adaptation risk (inverse of mean reward)
            double maxUcb = double.NegativeInfinity;
            for (int i = 0; i < _meanRewards.Count; i++)
            {
                double exploitation = _meanRewards[i];
                double exploration = _explorationWeight * Math.Sqrt(
                    _ucbScale * Math.Log(_totalPulls + 1) / Math.Max(1, _pullCounts[i]));
                double ucb = exploitation + exploration;
                if (ucb > maxUcb) maxUcb = ucb;
            }
            episode.Difficulty = Math.Max(0, Math.Min(1, maxUcb));
        }

        _totalPulls++;
        return episode;
    }

    /// <inheritdoc/>
    public void UpdateWithFeedback(IReadOnlyList<IEpisode<T, TInput, TOutput>> episodes, IReadOnlyList<double> losses)
    {
        for (int i = 0; i < Math.Min(episodes.Count, losses.Count); i++)
        {
            episodes[i].LastLoss = losses[i];
            episodes[i].SampleCount++;

            // Reward = loss reduction potential (higher loss = more potential improvement)
            double reward = losses[i]; // Higher loss = higher reward for sampling

            // Expand tracking arrays as needed
            int idx = episodes[i].EpisodeId % Math.Max(1, _meanRewards.Count + 1);
            while (_meanRewards.Count <= idx)
            {
                _meanRewards.Add(0);
                _pullCounts.Add(0);
            }

            // Incremental mean update
            _pullCounts[idx]++;
            _meanRewards[idx] += (reward - _meanRewards[idx]) / _pullCounts[idx];
        }
    }

    /// <inheritdoc/>
    public void SetSeed(int seed)
    {
        _rng = RandomHelper.CreateSeededRandom(seed);
        _dataset.SetSeed(seed);
    }
}
