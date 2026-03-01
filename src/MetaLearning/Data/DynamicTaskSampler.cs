using AiDotNet.Data.Structures;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

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
        Guard.NotNull(dataset);
        if (numWays <= 0)
            throw new ArgumentOutOfRangeException(nameof(numWays), numWays, "Number of ways must be positive.");
        if (numShots <= 0)
            throw new ArgumentOutOfRangeException(nameof(numShots), numShots, "Number of shots must be positive.");
        if (numQueryPerClass <= 0)
            throw new ArgumentOutOfRangeException(nameof(numQueryPerClass), numQueryPerClass, "Number of query examples per class must be positive.");

        _dataset = dataset;
        NumWays = numWays;
        NumShots = numShots;
        NumQueryPerClass = numQueryPerClass;
        _explorationRate = Math.Max(0, Math.Min(1.0, explorationRate));
        _lossHistory = new List<double>();
        _rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc/>
    public TaskBatch<T, TInput, TOutput> SampleBatch(int batchSize)
    {
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "Batch size must be positive.");

        var episodes = new List<IEpisode<T, TInput, TOutput>>(batchSize);
        var tasks = new IMetaLearningTask<T, TInput, TOutput>[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            var ep = SampleOne();
            episodes.Add(ep);
            tasks[i] = ep.Task;
        }

        // Assign per-episode difficulty scores that vary based on individual episode losses
        T[]? difficulties = null;
        if (_feedbackCount > 0)
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            difficulties = new T[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                // Use per-episode difficulty (set by SampleOne from loss history)
                double rawDifficulty = episodes[i].Difficulty ?? _runningMeanLoss;
                double d = Math.Max(0.0, Math.Min(1.0, rawDifficulty));
                difficulties[i] = numOps.FromDouble(d);
            }
        }

        return new TaskBatch<T, TInput, TOutput>(tasks, BatchingStrategy.Adaptive, difficulties);
    }

    /// <inheritdoc/>
    public IEpisode<T, TInput, TOutput> SampleOne()
    {
        var episode = _dataset.SampleEpisode(NumWays, NumShots, NumQueryPerClass);

        if (_feedbackCount > 0 && _rng.NextDouble() > _explorationRate)
        {
            // Sample difficulty from recent loss distribution (mean + noise from variance)
            // This produces per-episode variation rather than identical difficulty values
            double lossStdDev = 0;
            if (_lossHistory.Count > 1)
            {
                double sumSqDiff = 0;
                int recentStart = Math.Max(0, _lossHistory.Count - 100);
                int count = _lossHistory.Count - recentStart;
                for (int i = recentStart; i < _lossHistory.Count; i++)
                {
                    double diff = _lossHistory[i] - _runningMeanLoss;
                    sumSqDiff += diff * diff;
                }
                lossStdDev = Math.Sqrt(sumSqDiff / count);
            }

            // Difficulty = mean + random perturbation scaled by std dev, clamped to [0, 1]
            double noise = (_rng.NextDouble() - 0.5) * 2.0 * lossStdDev;
            episode.Difficulty = Math.Max(0, Math.Min(1.0, _runningMeanLoss + noise));
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
