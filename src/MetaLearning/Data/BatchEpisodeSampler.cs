using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Data;

/// <summary>
/// Efficiently samples batches of episodes with optional prefetching and caching.
/// Wraps an <see cref="ITaskSampler{T, TInput, TOutput}"/> and provides batch-level operations
/// such as prefetching the next batch while the current batch is being trained on.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public class BatchEpisodeSampler<T, TInput, TOutput>
{
    private readonly ITaskSampler<T, TInput, TOutput> _sampler;
    private readonly int _batchSize;
    private readonly int _prefetchCount;
    private readonly Queue<IEpisode<T, TInput, TOutput>> _prefetchBuffer;

    /// <summary>
    /// Gets the number of episodes per batch.
    /// </summary>
    public int BatchSize => _batchSize;

    /// <summary>
    /// Gets the number of episodes currently in the prefetch buffer.
    /// </summary>
    public int PrefetchedCount => _prefetchBuffer.Count;

    /// <summary>
    /// Creates a batch episode sampler.
    /// </summary>
    /// <param name="sampler">The underlying task sampler.</param>
    /// <param name="batchSize">Number of episodes per batch.</param>
    /// <param name="prefetchCount">Number of batches to prefetch. 0 means no prefetching.</param>
    public BatchEpisodeSampler(
        ITaskSampler<T, TInput, TOutput> sampler,
        int batchSize = 4,
        int prefetchCount = 0)
    {
        _sampler = sampler;
        _batchSize = Math.Max(1, batchSize);
        _prefetchCount = Math.Max(0, prefetchCount);
        _prefetchBuffer = new Queue<IEpisode<T, TInput, TOutput>>();

        // Pre-fill the buffer
        FillPrefetchBuffer();
    }

    /// <summary>
    /// Gets the next batch of episodes.
    /// </summary>
    /// <returns>A list of episodes forming one batch.</returns>
    public IReadOnlyList<IEpisode<T, TInput, TOutput>> NextBatch()
    {
        var batch = new List<IEpisode<T, TInput, TOutput>>(_batchSize);

        for (int i = 0; i < _batchSize; i++)
        {
            if (_prefetchBuffer.Count > 0)
            {
                batch.Add(_prefetchBuffer.Dequeue());
            }
            else
            {
                batch.Add(_sampler.SampleOne());
            }
        }

        // Refill buffer after consuming
        FillPrefetchBuffer();
        return batch;
    }

    /// <summary>
    /// Gets a task batch suitable for <see cref="MetaLearnerBase{T, TInput, TOutput}.MetaTrain"/>.
    /// </summary>
    /// <returns>A TaskBatch of the configured batch size.</returns>
    public TaskBatch<T, TInput, TOutput> NextTaskBatch()
    {
        return _sampler.SampleBatch(_batchSize);
    }

    /// <summary>
    /// Provides feedback for the sampled episodes (delegates to underlying sampler).
    /// </summary>
    /// <param name="episodes">The episodes that were evaluated.</param>
    /// <param name="losses">The loss for each episode.</param>
    public void ProvideFeedback(IReadOnlyList<IEpisode<T, TInput, TOutput>> episodes, IReadOnlyList<double> losses)
    {
        _sampler.UpdateWithFeedback(episodes, losses);
    }

    private void FillPrefetchBuffer()
    {
        int target = _prefetchCount * _batchSize;
        while (_prefetchBuffer.Count < target)
        {
            _prefetchBuffer.Enqueue(_sampler.SampleOne());
        }
    }
}
