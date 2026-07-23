using AiDotNet.Data.Structures;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for data loaders that provide episodic tasks for meta-learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The input data type for tasks.</typeparam>
/// <typeparam name="TOutput">The output data type for tasks.</typeparam>
/// <remarks>
/// <para>
/// This interface is for meta-learning scenarios using N-way K-shot learning,
/// where the loader generates tasks consisting of:
/// - Support set: K examples per class for N classes (used to adapt the model)
/// - Query set: Additional examples for evaluation after adaptation
/// </para>
/// <para><b>For Beginners:</b> Meta-learning is "learning to learn".
///
/// **Standard ML**: Train on lots of cat/dog images, then classify new cat/dog images.
///
/// **Meta-learning**: Train on many different tasks (cats vs dogs, cars vs planes, etc.),
/// then when given a *new* task with only a few examples, quickly learn to do it.
///
/// **N-way K-shot** means:
/// - **N-way**: Each task has N different classes to distinguish
/// - **K-shot**: You get K examples of each class to learn from
///
/// **Example: 5-way 1-shot**
/// - Given 5 new animal types you've never seen
/// - With only 1 example image of each
/// - Classify new images into one of these 5 types
///
/// The episodic data loader creates these mini-tasks for training.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("EpisodicDataLoader")]
public interface IEpisodicDataLoader<T, TInput, TOutput> :
    IDataLoader<T>,
    IBatchIterable<MetaLearningTask<T, TInput, TOutput>>
{
    /// <summary>
    /// Gets the number of classes per task (N in N-way).
    /// </summary>
    int NWay { get; }

    /// <summary>
    /// Gets the number of support examples per class (K in K-shot).
    /// </summary>
    int KShot { get; }

    /// <summary>
    /// Gets the number of query examples per class.
    /// </summary>
    int QueryShots { get; }

    /// <summary>
    /// Gets the total number of available classes in the dataset.
    /// </summary>
    int AvailableClasses { get; }

    /// <summary>
    /// Gets the next meta-learning task (support set + query set).
    /// </summary>
    /// <returns>A MetaLearningTask with support and query sets.</returns>
    /// <remarks>
    /// <para>
    /// Each call returns a new randomly sampled task with:
    /// - N randomly selected classes from available classes
    /// - K support examples per class
    /// - QueryShots query examples per class
    /// </para>
    /// </remarks>
    MetaLearningTask<T, TInput, TOutput> GetNextTask();

    /// <summary>
    /// Gets multiple meta-learning tasks as a batch.
    /// </summary>
    /// <param name="numTasks">Number of tasks to sample.</param>
    /// <returns>A list of MetaLearningTasks.</returns>
    IReadOnlyList<MetaLearningTask<T, TInput, TOutput>> GetTaskBatch(int numTasks);

    /// <summary>
    /// Sets the random seed for reproducible task sampling.
    /// </summary>
    /// <param name="seed">Random seed value.</param>
    void SetSeed(int seed);
}
