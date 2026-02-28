using AiDotNet.Data.Structures;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Data;

/// <summary>
/// Samples tasks uniformly at random from a meta-dataset.
/// This is the simplest and most commonly used sampling strategy.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public class UniformTaskSampler<T, TInput, TOutput> : ITaskSampler<T, TInput, TOutput>
{
    private readonly IMetaDataset<T, TInput, TOutput> _dataset;

    /// <inheritdoc/>
    public int NumWays { get; }

    /// <inheritdoc/>
    public int NumShots { get; }

    /// <inheritdoc/>
    public int NumQueryPerClass { get; }

    /// <summary>
    /// Creates a uniform task sampler over the given meta-dataset.
    /// </summary>
    /// <param name="dataset">The meta-dataset to sample from.</param>
    /// <param name="numWays">Number of classes per task.</param>
    /// <param name="numShots">Number of support examples per class.</param>
    /// <param name="numQueryPerClass">Number of query examples per class.</param>
    public UniformTaskSampler(
        IMetaDataset<T, TInput, TOutput> dataset,
        int numWays = 5,
        int numShots = 1,
        int numQueryPerClass = 15)
    {
        _dataset = dataset;
        NumWays = numWays;
        NumShots = numShots;
        NumQueryPerClass = numQueryPerClass;
    }

    /// <inheritdoc/>
    public TaskBatch<T, TInput, TOutput> SampleBatch(int batchSize)
    {
        var episodes = _dataset.SampleEpisodes(batchSize, NumWays, NumShots, NumQueryPerClass);
        var tasks = new IMetaLearningTask<T, TInput, TOutput>[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            tasks[i] = episodes[i].Task;
        }
        return new TaskBatch<T, TInput, TOutput>(tasks, BatchingStrategy.Uniform);
    }

    /// <inheritdoc/>
    public IEpisode<T, TInput, TOutput> SampleOne()
    {
        return _dataset.SampleEpisode(NumWays, NumShots, NumQueryPerClass);
    }

    /// <inheritdoc/>
    public void UpdateWithFeedback(IReadOnlyList<IEpisode<T, TInput, TOutput>> episodes, IReadOnlyList<double> losses)
    {
        // Uniform sampler ignores feedback — sampling is always random.
    }

    /// <inheritdoc/>
    public void SetSeed(int seed)
    {
        _dataset.SetSeed(seed);
    }
}
