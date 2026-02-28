using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Data;

/// <summary>
/// Adapter that wraps an existing <see cref="IEpisodicDataLoader{T, TInput, TOutput}"/>
/// to implement the <see cref="ITaskSampler{T, TInput, TOutput}"/> interface.
/// This provides backward compatibility so that legacy data loaders can be used
/// with the new task-sampler-based infrastructure.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public class EpisodicDataLoaderTaskSamplerAdapter<T, TInput, TOutput> : ITaskSampler<T, TInput, TOutput>
{
    private readonly IEpisodicDataLoader<T, TInput, TOutput> _loader;

    /// <inheritdoc/>
    public int NumWays => _loader.NWay;

    /// <inheritdoc/>
    public int NumShots => _loader.KShot;

    /// <inheritdoc/>
    public int NumQueryPerClass => _loader.QueryShots;

    /// <summary>
    /// Creates an adapter wrapping the given episodic data loader.
    /// </summary>
    /// <param name="loader">The legacy episodic data loader to adapt.</param>
    public EpisodicDataLoaderTaskSamplerAdapter(IEpisodicDataLoader<T, TInput, TOutput> loader)
    {
        _loader = loader;
    }

    /// <inheritdoc/>
    public TaskBatch<T, TInput, TOutput> SampleBatch(int batchSize)
    {
        var loaderTasks = _loader.GetTaskBatch(batchSize);
        var tasks = new IMetaLearningTask<T, TInput, TOutput>[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            tasks[i] = loaderTasks[i];
        }
        return new TaskBatch<T, TInput, TOutput>(tasks, BatchingStrategy.Uniform);
    }

    /// <inheritdoc/>
    public IEpisode<T, TInput, TOutput> SampleOne()
    {
        var task = _loader.GetNextTask();
        return new Episode<T, TInput, TOutput>(task);
    }

    /// <inheritdoc/>
    public void UpdateWithFeedback(IReadOnlyList<IEpisode<T, TInput, TOutput>> episodes, IReadOnlyList<double> losses)
    {
        // Legacy data loaders don't support feedback — this is a no-op.
    }

    /// <inheritdoc/>
    public void SetSeed(int seed)
    {
        _loader.SetSeed(seed);
    }
}
