using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Data;

/// <summary>
/// Concrete implementation of <see cref="IEpisode{T, TInput, TOutput}"/> that wraps a
/// meta-learning task with episode-level metadata.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public class Episode<T, TInput, TOutput> : IEpisode<T, TInput, TOutput>
{
    private static int _nextId;

    /// <inheritdoc/>
    public IMetaLearningTask<T, TInput, TOutput> Task { get; }

    /// <inheritdoc/>
    public int EpisodeId { get; }

    /// <inheritdoc/>
    public string? Domain { get; }

    /// <inheritdoc/>
    public double? Difficulty { get; set; }

    /// <inheritdoc/>
    public double? LastLoss { get; set; }

    /// <inheritdoc/>
    public long CreatedTimestamp { get; }

    /// <inheritdoc/>
    public int SampleCount { get; set; }

    /// <inheritdoc/>
    public Dictionary<string, object>? EpisodeMetadata { get; }

    /// <summary>
    /// Creates a new episode wrapping the given task.
    /// </summary>
    /// <param name="task">The meta-learning task for this episode.</param>
    /// <param name="domain">Optional domain label.</param>
    /// <param name="difficulty">Optional difficulty score in [0, 1].</param>
    /// <param name="metadata">Optional key-value metadata.</param>
    public Episode(
        IMetaLearningTask<T, TInput, TOutput> task,
        string? domain = null,
        double? difficulty = null,
        Dictionary<string, object>? metadata = null)
    {
        Task = task;
        EpisodeId = System.Threading.Interlocked.Increment(ref _nextId);
        Domain = domain;
        Difficulty = difficulty;
        CreatedTimestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        EpisodeMetadata = metadata;
    }
}
