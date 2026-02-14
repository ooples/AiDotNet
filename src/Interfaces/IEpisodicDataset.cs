using System.Collections.Generic;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for episodic datasets used in meta-learning.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
[AiDotNet.Configuration.YamlConfigurable("EpisodicDataset")]
public interface IEpisodicDataset<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the total number of episodes in the dataset.
    /// </summary>
    int EpisodeCount { get; }

    /// <summary>
    /// Gets the number of classes per episode (N-way).
    /// </summary>
    int ClassesPerEpisode { get; }

    /// <summary>
    /// Gets the number of examples per class (K-shot).
    /// </summary>
    int ExamplesPerClass { get; }

    /// <summary>
    /// Gets the number of query examples per class.
    /// </summary>
    int QueryExamplesPerClass { get; }

    /// <summary>
    /// Gets a specific episode from the dataset.
    /// </summary>
    /// <param name="episodeIndex">The index of the episode to retrieve.</param>
    /// <returns>The episode containing support and query sets.</returns>
    IMetaLearningTask<T, TInput, TOutput> GetEpisode(int episodeIndex);

    /// <summary>
    /// Gets a batch of episodes from the dataset.
    /// </summary>
    /// <param name="batchSize">The number of episodes to include in the batch.</param>
    /// <param name="shuffle">Whether to shuffle the episodes before sampling.</param>
    /// <returns>A batch of episodes.</returns>
    List<IMetaLearningTask<T, TInput, TOutput>> GetEpisodeBatch(int batchSize, bool shuffle = true);

    /// <summary>
    /// Shuffles the dataset.
    /// </summary>
    void Shuffle();

    /// <summary>
    /// Resets the dataset to the beginning.
    /// </summary>
    void Reset();

    /// <summary>
    /// Gets whether the dataset has more episodes.
    /// </summary>
    bool HasMoreEpisodes { get; }

    /// <summary>
    /// Sets the random seed for reproducible sampling.
    /// </summary>
    /// <param name="seed">The random seed to use.</param>
    void SetRandomSeed(int seed);

    /// <summary>
    /// Samples a batch of tasks from the dataset.
    /// </summary>
    /// <param name="numTasks">The number of tasks to sample.</param>
    /// <param name="episodeIndex">The starting episode index (optional).</param>
    /// <returns>A list of sampled tasks.</returns>
    List<IMetaLearningTask<T, TInput, TOutput>> SampleTasks(int numTasks, int? episodeIndex = null);
}
