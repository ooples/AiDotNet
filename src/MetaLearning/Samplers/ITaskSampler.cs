using AiDotNet.MetaLearning.Datasets;
using AiDotNet.MetaLearning.Tasks;

namespace AiDotNet.MetaLearning.Samplers;

/// <summary>
/// Interface for sampling episodic tasks from meta-learning datasets.
/// </summary>
/// <typeparam name="T">The numeric type (float, double)</typeparam>
/// <remarks>
/// For Beginners:
/// A task sampler creates N-way K-shot episodes by:
/// 1. Randomly selecting N classes from the dataset
/// 2. Sampling K examples per class for the support set
/// 3. Sampling Q examples per class for the query set
/// This process creates mini-tasks that the model learns to solve quickly.
/// </remarks>
public interface ITaskSampler<T> where T : struct
{
    /// <summary>
    /// Gets the number of ways (classes per episode).
    /// </summary>
    int NumWays { get; }

    /// <summary>
    /// Gets the number of shots (support examples per class).
    /// </summary>
    int NumShots { get; }

    /// <summary>
    /// Gets the number of queries (query examples per class).
    /// </summary>
    int NumQueries { get; }

    /// <summary>
    /// Samples a single episode from the dataset.
    /// </summary>
    /// <returns>A new episode with support and query sets</returns>
    IEpisode<T> SampleEpisode();

    /// <summary>
    /// Samples a batch of episodes.
    /// </summary>
    /// <param name="batchSize">Number of episodes to sample</param>
    /// <returns>Array of episodes</returns>
    IEpisode<T>[] SampleBatch(int batchSize);

    /// <summary>
    /// Sets the random seed for reproducibility.
    /// </summary>
    /// <param name="seed">Random seed</param>
    void SetSeed(int seed);

    /// <summary>
    /// Gets the underlying dataset.
    /// </summary>
    IMetaDataset<T> Dataset { get; }
}
