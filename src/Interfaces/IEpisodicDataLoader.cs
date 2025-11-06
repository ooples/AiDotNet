using AiDotNet.Data.Abstractions;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for episodic data loaders that sample N-way K-shot meta-learning tasks.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This interface defines the core functionality required for meta-learning data loaders.
/// Implementations must be able to generate episodic tasks suitable for few-shot learning
/// algorithms such as MAML, Reptile, and SEAL.
/// </para>
/// <para><b>For Beginners:</b> This interface describes what any meta-learning data loader must do.
///
/// Meta-learning requires training on many small tasks instead of one large dataset. This interface
/// ensures that any data loader can provide these tasks in a consistent way, making it easy to:
/// - Swap between different data loading strategies
/// - Use dependency injection in your application
/// - Mock data loaders for testing
/// - Extend functionality with custom implementations
///
/// The key requirement is the ability to generate tasks on demand, where each task contains
/// a support set (for learning) and a query set (for evaluation).
/// </para>
/// </remarks>
public interface IEpisodicDataLoader<T>
{
    /// <summary>
    /// Samples and returns the next N-way K-shot meta-learning task.
    /// </summary>
    /// <returns>
    /// A MetaLearningTask containing support and query sets with the configured N-way K-shot specification.
    /// </returns>
    /// <remarks>
    /// <para>
    /// Each call to this method should produce a new task with randomly sampled classes and examples.
    /// The task structure must conform to the N-way K-shot configuration specified during initialization.
    /// </para>
    /// <para><b>For Beginners:</b> This method generates one meta-learning task each time you call it.
    ///
    /// Think of it like drawing a random quiz from a question bank:
    /// - Each quiz (task) tests different topics (classes)
    /// - Each quiz has some practice questions (support set) and test questions (query set)
    /// - Every time you call this method, you get a new, different quiz
    ///
    /// The meta-learning algorithm will train on many such tasks, learning to quickly
    /// adapt to new tasks by recognizing patterns across diverse problems.
    /// </para>
    /// </remarks>
    MetaLearningTask<T> GetNextTask();
}
