namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a single episode in meta-learning, wrapping an <see cref="IMetaLearningTask{T, TInput, TOutput}"/>
/// with additional metadata such as domain, difficulty, and timing information.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> An episode is one complete "mini-learning session" in meta-learning.
/// Each episode contains a task (support set + query set) along with extra information about that task,
/// like how hard it is and what domain it came from.
/// </para>
/// </remarks>
public interface IEpisode<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the underlying meta-learning task containing support and query sets.
    /// </summary>
    IMetaLearningTask<T, TInput, TOutput> Task { get; }

    /// <summary>
    /// Gets the unique identifier for this episode.
    /// </summary>
    int EpisodeId { get; }

    /// <summary>
    /// Gets an optional domain or category label for this episode.
    /// </summary>
    /// <remarks>
    /// Useful for cross-domain meta-learning where episodes come from different data distributions.
    /// </remarks>
    string? Domain { get; }

    /// <summary>
    /// Gets an optional difficulty score for this episode, typically in [0, 1].
    /// </summary>
    /// <remarks>
    /// Used by curriculum-based samplers to order episodes from easy to hard during training.
    /// </remarks>
    double? Difficulty { get; set; }

    /// <summary>
    /// Gets or sets the loss observed on this episode during the most recent evaluation.
    /// </summary>
    /// <remarks>
    /// Tracked so that dynamic samplers can prioritize high-loss episodes.
    /// </remarks>
    double? LastLoss { get; set; }

    /// <summary>
    /// Gets the timestamp when this episode was created.
    /// </summary>
    long CreatedTimestamp { get; }

    /// <summary>
    /// Gets or sets the number of times this episode has been sampled.
    /// </summary>
    int SampleCount { get; set; }

    /// <summary>
    /// Gets optional key-value metadata associated with the episode.
    /// </summary>
    Dictionary<string, object>? EpisodeMetadata { get; }
}
