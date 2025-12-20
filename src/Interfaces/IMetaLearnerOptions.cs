namespace AiDotNet.Interfaces;

/// <summary>
/// Configuration options interface for meta-learning algorithms.
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
/// - <b>Inner loop:</b> How you study for a specific exam (practice problems, examples)
/// - <b>Outer loop:</b> Learning better study techniques by reflecting across many exams
///
/// The configuration controls both loops:
/// - InnerLearningRate: How aggressively to adapt to each task
/// - OuterLearningRate: How much to update the meta-parameters
/// - AdaptationSteps: How many gradient updates per task
/// </para>
/// </remarks>
public interface IMetaLearnerOptions<T>
{
    /// <summary>
    /// Gets the inner loop learning rate for task-specific adaptation.
    /// </summary>
    /// <value>
    /// The learning rate used during task adaptation on support sets.
    /// Typical values: 0.001 to 0.1. Default: 0.01
    /// </value>
    double InnerLearningRate { get; }

    /// <summary>
    /// Gets the outer loop learning rate for meta-optimization.
    /// </summary>
    /// <value>
    /// The learning rate for updating meta-parameters.
    /// Typically 10x smaller than InnerLearningRate.
    /// Typical values: 0.0001 to 0.01. Default: 0.001
    /// </value>
    double OuterLearningRate { get; }

    /// <summary>
    /// Gets the number of gradient descent steps for inner loop adaptation.
    /// </summary>
    /// <value>
    /// How many times to update parameters on each task's support set.
    /// Typical values: 1 to 10. Default: 5
    /// </value>
    int AdaptationSteps { get; }

    /// <summary>
    /// Gets the number of tasks to sample per meta-update (meta-batch size).
    /// </summary>
    /// <value>
    /// How many tasks to average over for each outer loop update.
    /// Typical values: 1 (online) to 32 (batch). Default: 4
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
    /// Gets the number of meta-training iterations to perform.
    /// </summary>
    /// <value>
    /// How many times to perform the outer loop meta-update.
    /// Typical values: 100 to 10,000. Default: 1000
    /// </value>
    int NumMetaIterations { get; }

    /// <summary>
    /// Gets whether to use first-order approximation (e.g., FOMAML, Reptile).
    /// </summary>
    /// <value>
    /// True to ignore second-order gradients, which is faster but may be less accurate.
    /// Default: false for MAML-based algorithms.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> First-order approximation:
    /// - True: Faster training, simpler gradients, works well in practice
    /// - False: More accurate gradients, slower, may be unstable with many inner steps
    /// </para>
    /// </remarks>
    bool UseFirstOrder { get; }

    /// <summary>
    /// Gets the gradient clipping threshold to prevent exploding gradients.
    /// </summary>
    /// <value>
    /// Maximum gradient norm. Set to null or 0 to disable clipping.
    /// Typical values: 1.0 to 10.0. Default: 10.0
    /// </value>
    double? GradientClipThreshold { get; }

    /// <summary>
    /// Gets the random seed for reproducible task sampling and initialization.
    /// </summary>
    /// <value>
    /// Random seed value. Set to null for non-deterministic behavior.
    /// </value>
    int? RandomSeed { get; }

    /// <summary>
    /// Gets the number of evaluation tasks for periodic validation.
    /// </summary>
    /// <value>
    /// How many tasks to use when evaluating model performance.
    /// Typical values: 100 to 1000. Default: 100
    /// </value>
    int EvaluationTasks { get; }

    /// <summary>
    /// Gets the evaluation frequency in meta-iterations.
    /// </summary>
    /// <value>
    /// Run evaluation every N meta-iterations. Set to 0 to disable periodic evaluation.
    /// Default: 100
    /// </value>
    int EvaluationFrequency { get; }

    /// <summary>
    /// Gets whether to save checkpoints during training.
    /// </summary>
    /// <value>
    /// True to save model checkpoints periodically.
    /// </value>
    bool EnableCheckpointing { get; }

    /// <summary>
    /// Gets the checkpoint save frequency in meta-iterations.
    /// </summary>
    /// <value>
    /// Save checkpoint every N meta-iterations. Only used if EnableCheckpointing is true.
    /// Default: 500
    /// </value>
    int CheckpointFrequency { get; }

    /// <summary>
    /// Validates that the configuration is valid and sensible.
    /// </summary>
    /// <returns>True if the configuration is valid; false otherwise.</returns>
    bool IsValid();

    /// <summary>
    /// Creates a deep copy of this options instance.
    /// </summary>
    /// <returns>A new options instance with the same values.</returns>
    IMetaLearnerOptions<T> Clone();
}
