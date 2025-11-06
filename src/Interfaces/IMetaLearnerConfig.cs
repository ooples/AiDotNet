namespace AiDotNet.Interfaces;

/// <summary>
/// Configuration interface for meta-learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric type (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Meta-learning algorithms use a two-loop optimization structure:
/// - <b>Inner loop:</b> Fast adaptation to a specific task using support set
/// - <b>Outer loop:</b> Meta-optimization to improve adaptation across all tasks
/// </para>
/// <para><b>For Beginners:</b> Think of meta-learning like learning to study effectively:
///
/// - <b>Inner loop:</b> How you study for a specific exam (cramming, practice problems)
/// - <b>Outer loop:</b> Learning better study techniques by reflecting across many exams
///
/// The configuration controls both loops:
/// - InnerLearningRate: How aggressively to adapt to each task
/// - MetaLearningRate: How much to update the meta-parameters based on task adaptations
/// - InnerSteps: How many gradient updates per task
/// </para>
/// </remarks>
public interface IMetaLearnerConfig<T>
{
    /// <summary>
    /// Gets the inner loop learning rate for task-specific adaptation.
    /// </summary>
    /// <value>
    /// The learning rate used during task adaptation on support sets.
    /// Typical values: 0.001 to 0.1
    /// </value>
    T InnerLearningRate { get; }

    /// <summary>
    /// Gets the outer loop learning rate for meta-optimization (meta-learning rate).
    /// </summary>
    /// <value>
    /// The learning rate for updating meta-parameters.
    /// Typically 10x smaller than InnerLearningRate.
    /// Typical values: 0.0001 to 0.01
    /// </value>
    T MetaLearningRate { get; }

    /// <summary>
    /// Gets the number of gradient descent steps for inner loop adaptation.
    /// </summary>
    /// <value>
    /// How many times to update parameters on each task's support set.
    /// Typical values: 1 to 10
    /// </value>
    int InnerSteps { get; }

    /// <summary>
    /// Gets the number of tasks to sample per meta-update (meta-batch size).
    /// </summary>
    /// <value>
    /// How many tasks to average over for each outer loop update.
    /// Typical values: 1 (online) to 32 (batch)
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls how many tasks you learn from before updating
    /// your meta-parameters:
    /// - MetaBatchSize = 1: Update after every task (more noisy, faster iteration)
    /// - MetaBatchSize = 16: Update after 16 tasks (more stable, slower iteration)
    /// </para>
    /// </remarks>
    int MetaBatchSize { get; }

    /// <summary>
    /// Validates that the configuration is valid and sensible.
    /// </summary>
    /// <returns>True if the configuration is valid; false otherwise.</returns>
    bool IsValid();
}
