using AiDotNet.Interfaces;
using AiDotNet.MetaLearning.Data;
using AiDotNet.Models;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Represents a meta-learning algorithm that can learn from multiple tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<T>, Tensor<T>).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Meta-learning is "learning to learn" - the algorithm practices
/// adapting to new tasks quickly by training on many different tasks.
///
/// Think of it like learning to learn languages:
/// - Instead of just learning one language, you learn many languages
/// - Over time, you get better at picking up new languages quickly
/// - When you encounter a new language, you can learn it faster than the first time
///
/// Similarly, a meta-learning algorithm:
/// - Trains on many different tasks
/// - Learns patterns that help it adapt quickly to new tasks
/// - Can solve new tasks with just a few examples (few-shot learning)
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("MetaLearningAlgorithm")]
public interface IMetaLearningAlgorithm<T, TInput, TOutput>
{
    /// <summary>
    /// Performs one meta-training step on a batch of tasks.
    /// </summary>
    /// <param name="taskBatch">The batch of tasks to train on.</param>
    /// <returns>The meta-training loss for this batch.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method updates the model by training on multiple tasks at once.
    /// Each task teaches the model something about how to learn quickly. The returned loss value
    /// indicates how well the model is doing - lower is better.
    /// </para>
    /// </remarks>
    T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch);

    /// <summary>
    /// Adapts the model to a new task using its support set.
    /// </summary>
    /// <param name="task">The task to adapt to.</param>
    /// <returns>A new model instance adapted to the task.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is where the "quick learning" happens. Given a new task
    /// with just a few examples (the support set), this method creates a new model that's
    /// specialized for that specific task. This is what makes meta-learning powerful -
    /// it can adapt to new tasks with very few examples.
    /// </para>
    /// </remarks>
    IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task);

    /// <summary>
    /// Evaluates the meta-learning algorithm on a batch of tasks.
    /// </summary>
    /// <param name="taskBatch">The batch of tasks to evaluate on.</param>
    /// <returns>The average evaluation loss across all tasks.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This checks how well the meta-learning algorithm performs.
    /// For each task, it adapts using the support set and then tests on the query set.
    /// The returned value is the average loss across all tasks - lower means better performance.
    /// </para>
    /// </remarks>
    T Evaluate(TaskBatch<T, TInput, TOutput> taskBatch);

    /// <summary>
    /// Gets the base model used by this meta-learning algorithm.
    /// </summary>
    /// <returns>The base model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns the "meta-learned" model that has been trained
    /// on many tasks. This model itself may not be very good at any specific task, but it's
    /// excellent as a starting point for quickly adapting to new tasks.
    /// </para>
    /// </remarks>
    IFullModel<T, TInput, TOutput> GetMetaModel();

    /// <summary>
    /// Sets the base model for this meta-learning algorithm.
    /// </summary>
    /// <param name="model">The model to use as the base.</param>
    void SetMetaModel(IFullModel<T, TInput, TOutput> model);

    /// <summary>
    /// Gets the name of the meta-learning algorithm.
    /// </summary>
    string AlgorithmName { get; }

    /// <summary>
    /// Gets the number of adaptation steps to perform during task adaptation.
    /// </summary>
    int AdaptationSteps { get; }

    /// <summary>
    /// Gets the learning rate used for task adaptation (inner loop).
    /// </summary>
    double InnerLearningRate { get; }

    /// <summary>
    /// Gets the learning rate used for meta-learning (outer loop).
    /// </summary>
    double OuterLearningRate { get; }
}
