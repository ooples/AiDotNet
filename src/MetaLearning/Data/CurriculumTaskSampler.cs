using AiDotNet.Data.Structures;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Data;

/// <summary>
/// Samples tasks following a difficulty-based curriculum: starts with easy tasks and
/// gradually increases difficulty as training progresses. Uses episode difficulty scores
/// and observed losses to order task presentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Just like teaching a student easy problems before hard ones,
/// curriculum learning presents the meta-learner with simple tasks first and gradually
/// introduces harder tasks. This typically leads to faster and more stable training.</para>
/// </remarks>
public class CurriculumTaskSampler<T, TInput, TOutput> : ITaskSampler<T, TInput, TOutput>
{
    private readonly IMetaDataset<T, TInput, TOutput> _dataset;
    private readonly double _paceRate;
    private double _currentDifficulty;
    private int _totalSteps;
    private Random _rng;

    /// <inheritdoc/>
    public int NumWays { get; }

    /// <inheritdoc/>
    public int NumShots { get; }

    /// <inheritdoc/>
    public int NumQueryPerClass { get; }

    /// <summary>
    /// Creates a curriculum task sampler that increases difficulty over time.
    /// </summary>
    /// <param name="dataset">The meta-dataset to sample from.</param>
    /// <param name="numWays">Number of classes per task.</param>
    /// <param name="numShots">Number of support examples per class.</param>
    /// <param name="numQueryPerClass">Number of query examples per class.</param>
    /// <param name="paceRate">How fast difficulty increases per feedback step.
    /// A value of 0.01 means difficulty increases by ~1% per step. Default: 0.01.</param>
    /// <param name="initialDifficulty">Starting difficulty in [0, 1]. Default: 0.1 (easy).</param>
    /// <param name="seed">Optional random seed.</param>
    public CurriculumTaskSampler(
        IMetaDataset<T, TInput, TOutput> dataset,
        int numWays = 5,
        int numShots = 1,
        int numQueryPerClass = 15,
        double paceRate = 0.01,
        double initialDifficulty = 0.1,
        int? seed = null)
    {
        _dataset = dataset;
        NumWays = numWays;
        NumShots = numShots;
        NumQueryPerClass = numQueryPerClass;
        _paceRate = paceRate;
        _currentDifficulty = Math.Max(0, Math.Min(1, initialDifficulty));
        _rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc/>
    public TaskBatch<T, TInput, TOutput> SampleBatch(int batchSize)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var tasks = new IMetaLearningTask<T, TInput, TOutput>[batchSize];
        var difficulties = new T[batchSize];

        for (int i = 0; i < batchSize; i++)
        {
            var episode = SampleOne();
            tasks[i] = episode.Task;
            difficulties[i] = numOps.FromDouble(episode.Difficulty ?? _currentDifficulty);
        }

        var stage = _currentDifficulty < 0.33 ? CurriculumStage.Easy
            : _currentDifficulty < 0.66 ? CurriculumStage.Medium
            : CurriculumStage.Hard;

        return new TaskBatch<T, TInput, TOutput>(tasks, BatchingStrategy.CurriculumAware, difficulties, curriculumStage: stage);
    }

    /// <inheritdoc/>
    public IEpisode<T, TInput, TOutput> SampleOne()
    {
        var episode = _dataset.SampleEpisode(NumWays, NumShots, NumQueryPerClass);
        // Assign current curriculum difficulty; add small noise for variety
        double noise = (_rng.NextDouble() - 0.5) * 0.1;
        episode.Difficulty = Math.Max(0, Math.Min(1, _currentDifficulty + noise));
        return episode;
    }

    /// <inheritdoc/>
    public void UpdateWithFeedback(IReadOnlyList<IEpisode<T, TInput, TOutput>> episodes, IReadOnlyList<double> losses)
    {
        for (int i = 0; i < Math.Min(episodes.Count, losses.Count); i++)
        {
            episodes[i].LastLoss = losses[i];
            episodes[i].SampleCount++;
        }

        _totalSteps++;

        // Advance difficulty: if average loss is low, increase faster; if high, slow down
        double avgLoss = 0;
        int count = Math.Min(episodes.Count, losses.Count);
        if (count > 0)
        {
            for (int i = 0; i < count; i++) avgLoss += losses[i];
            avgLoss /= count;
        }

        // Sigmoid-paced curriculum: difficulty increases faster when loss is low
        double progressRate = _paceRate / (1.0 + Math.Exp(avgLoss - 1.0));
        _currentDifficulty = Math.Min(1.0, _currentDifficulty + progressRate);
    }

    /// <inheritdoc/>
    public void SetSeed(int seed)
    {
        _rng = RandomHelper.CreateSeededRandom(seed);
        _dataset.SetSeed(seed);
    }
}
