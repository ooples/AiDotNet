namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a high-level meta-dataset that can generate episodes for meta-learning.
/// Unlike <see cref="IEpisodicDataset{T, TInput, TOutput}"/> which operates on pre-built episodes,
/// this interface generates episodes on-the-fly from an underlying data source.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A meta-dataset is a collection of data organized so that we can create
/// many different learning tasks from it. Each task (episode) contains a small training set (support)
/// and a small test set (query). The meta-learner trains across many such tasks to learn how to
/// learn quickly from small amounts of data.
/// </para>
/// </remarks>
public interface IMetaDataset<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the name of this meta-dataset.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the total number of distinct classes available in the dataset.
    /// </summary>
    int TotalClasses { get; }

    /// <summary>
    /// Gets the total number of examples across all classes.
    /// </summary>
    int TotalExamples { get; }

    /// <summary>
    /// Gets the number of examples available for each class, keyed by class index.
    /// </summary>
    IReadOnlyDictionary<int, int> ClassExampleCounts { get; }

    /// <summary>
    /// Samples a single episode from the dataset.
    /// </summary>
    /// <param name="numWays">Number of classes in the episode.</param>
    /// <param name="numShots">Number of support examples per class.</param>
    /// <param name="numQueryPerClass">Number of query examples per class.</param>
    /// <returns>An episode containing the sampled task.</returns>
    IEpisode<T, TInput, TOutput> SampleEpisode(int numWays, int numShots, int numQueryPerClass);

    /// <summary>
    /// Samples multiple episodes from the dataset.
    /// </summary>
    /// <param name="count">Number of episodes to sample.</param>
    /// <param name="numWays">Number of classes per episode.</param>
    /// <param name="numShots">Number of support examples per class.</param>
    /// <param name="numQueryPerClass">Number of query examples per class.</param>
    /// <returns>A list of sampled episodes.</returns>
    IReadOnlyList<IEpisode<T, TInput, TOutput>> SampleEpisodes(int count, int numWays, int numShots, int numQueryPerClass);

    /// <summary>
    /// Sets the random seed for reproducible episode generation.
    /// </summary>
    /// <param name="seed">The random seed value.</param>
    void SetSeed(int seed);

    /// <summary>
    /// Gets whether this dataset supports a given N-way K-shot configuration.
    /// </summary>
    /// <param name="numWays">Number of classes per episode.</param>
    /// <param name="numShots">Number of support examples per class.</param>
    /// <param name="numQueryPerClass">Number of query examples per class.</param>
    /// <returns>True if the configuration is feasible for this dataset.</returns>
    bool SupportsConfiguration(int numWays, int numShots, int numQueryPerClass);
}
