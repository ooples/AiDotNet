using AiDotNet.MetaLearning.Data;

namespace AiDotNet.Interfaces;

/// <summary>
/// Controls the strategy for sampling meta-learning tasks (episodes) from a dataset.
/// Implementations can provide uniform, balanced, curriculum-based, or dynamic sampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A task sampler decides <em>which</em> tasks (episodes) to give
/// the meta-learner during training. Different strategies can make training faster or more robust:
/// <list type="bullet">
/// <item><b>Uniform</b>: Picks tasks completely at random — simple and effective.</item>
/// <item><b>Balanced</b>: Ensures every class appears equally often across tasks.</item>
/// <item><b>Dynamic</b>: Focuses on tasks the model struggles with the most.</item>
/// <item><b>Curriculum</b>: Starts with easy tasks, gradually increases difficulty.</item>
/// </list>
/// </para>
/// </remarks>
public interface ITaskSampler<T, TInput, TOutput>
{
    /// <summary>
    /// Samples a batch of tasks from the underlying meta-dataset.
    /// </summary>
    /// <param name="batchSize">Number of tasks to sample.</param>
    /// <returns>A task batch ready for meta-training.</returns>
    TaskBatch<T, TInput, TOutput> SampleBatch(int batchSize);

    /// <summary>
    /// Samples a single episode from the underlying meta-dataset.
    /// </summary>
    /// <returns>A single sampled episode.</returns>
    IEpisode<T, TInput, TOutput> SampleOne();

    /// <summary>
    /// Updates the sampler state after observing a loss for a batch of tasks.
    /// </summary>
    /// <param name="episodes">The episodes that were evaluated.</param>
    /// <param name="losses">The loss observed for each episode.</param>
    /// <remarks>
    /// Dynamic and curriculum samplers use this feedback to adjust future sampling.
    /// Stateless samplers (e.g., uniform) may ignore this call.
    /// </remarks>
    void UpdateWithFeedback(IReadOnlyList<IEpisode<T, TInput, TOutput>> episodes, IReadOnlyList<double> losses);

    /// <summary>
    /// Sets the random seed for reproducible sampling.
    /// </summary>
    /// <param name="seed">The random seed.</param>
    void SetSeed(int seed);

    /// <summary>
    /// Gets the N-way configuration used by this sampler.
    /// </summary>
    int NumWays { get; }

    /// <summary>
    /// Gets the K-shot configuration used by this sampler.
    /// </summary>
    int NumShots { get; }

    /// <summary>
    /// Gets the number of query examples per class used by this sampler.
    /// </summary>
    int NumQueryPerClass { get; }
}
