using AiDotNet.Data.Structures;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Data;

/// <summary>
/// Samples tasks with probability proportional to the loss observed on previous evaluations.
/// Tasks with higher loss are sampled more frequently, focusing training on areas the model
/// finds most difficult (inspired by hard-example mining and curriculum learning research).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public class DynamicTaskSampler<T, TInput, TOutput> : ITaskSampler<T, TInput, TOutput>
{
    private readonly IMetaDataset<T, TInput, TOutput> _dataset;
    private readonly double _explorationRate;
    private readonly List<double> _lossHistory;
    private double _runningMeanLoss;
    private int _feedbackCount;
    private Random _rng;

    /// <inheritdoc/>
    public int NumWays { get; }

    /// <inheritdoc/>
    public int NumShots { get; }

    /// <inheritdoc/>
    public int NumQueryPerClass { get; }

    /// <summary>
    /// Creates a dynamic task sampler that adapts based on observed losses.
    /// </summary>
    /// <param name="dataset">The meta-dataset to sample from.</param>
    /// <param name="numWays">Number of classes per task.</param>
    /// <param name="numShots">Number of support examples per class.</param>
    /// <param name="numQueryPerClass">Number of query examples per class.</param>
    /// <param name="explorationRate">Probability of sampling uniformly instead of by loss.
    /// Keeps the sampler from getting stuck on a narrow set of tasks. Default: 0.1 (10%).</param>
    /// <param name="seed">Optional random seed.</param>
    public DynamicTaskSampler(
        IMetaDataset<T, TInput, TOutput> dataset,
        int numWays = 5,
        int numShots = 1,
        int numQueryPerClass = 15,
        double explorationRate = 0.1,
        int? seed = null)
    {
        _dataset = dataset;
        NumWays = numWays;
        NumShots = numShots;
        NumQueryPerClass = numQueryPerClass;
        _explorationRate = explorationRate;
        _lossHistory = new List<double>();
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

        // Assign difficulty scores based on running loss statistics
        T[]? difficulties = null;
        if (_feedbackCount > 0 && _runningMeanLoss > 0)
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            difficulties = new T[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                // Difficulty based on ratio to mean loss (clamped to [0, 1])
                double d = Math.Max(0, Math.Min(1.0, _runningMeanLoss));
                difficulties[i] = numOps.FromDouble(d);
            }
        }

        return new TaskBatch<T, TInput, TOutput>(tasks, BatchingStrategy.Adaptive, difficulties);
    }

    /// <inheritdoc/>
    public IEpisode<T, TInput, TOutput> SampleOne()
    {
        // Exploration-exploitation: with probability _explorationRate, sample uniformly
        var episode = _dataset.SampleEpisode(NumWays, NumShots, NumQueryPerClass);

        // If we have loss history and not exploring, set difficulty based on running mean
        if (_feedbackCount > 0 && _rng.NextDouble() > _explorationRate)
        {
            episode.Difficulty = Math.Max(0, Math.Min(1.0, _runningMeanLoss));
        }

        return episode;
    }

    /// <inheritdoc/>
    public void UpdateWithFeedback(IReadOnlyList<IEpisode<T, TInput, TOutput>> episodes, IReadOnlyList<double> losses)
    {
        for (int i = 0; i < Math.Min(episodes.Count, losses.Count); i++)
        {
            episodes[i].LastLoss = losses[i];
            episodes[i].SampleCount++;
            _lossHistory.Add(losses[i]);

            // Exponential moving average of loss
            _feedbackCount++;
            double alpha = Math.Min(1.0, 2.0 / (_feedbackCount + 1));
            _runningMeanLoss = alpha * losses[i] + (1 - alpha) * _runningMeanLoss;
        }

        // Keep history bounded
        if (_lossHistory.Count > 10000)
        {
            _lossHistory.RemoveRange(0, _lossHistory.Count - 5000);
        }
    }

    /// <inheritdoc/>
    public void SetSeed(int seed)
    {
        _rng = RandomHelper.CreateSeededRandom(seed);
        _dataset.SetSeed(seed);
    }
}
