namespace AiDotNet.ContinualLearning.Interfaces;

/// <summary>
/// Configuration interface for continual learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This interface defines the settings needed for continual learning,
/// such as learning rates, memory constraints, and regularization parameters.</para>
///
/// <para>Common parameters include:
/// - <b>Learning rate:</b> How much to update the model in each step
/// - <b>Memory size:</b> How many examples from previous tasks to remember
/// - <b>Regularization strength:</b> How strongly to preserve previous knowledge
/// - <b>Training iterations:</b> How many training steps per task
/// </para>
/// </remarks>
public interface IContinualLearnerConfig<T>
{
    /// <summary>
    /// The learning rate for training new tasks.
    /// </summary>
    T LearningRate { get; }

    /// <summary>
    /// Number of training epochs per task.
    /// </summary>
    int EpochsPerTask { get; }

    /// <summary>
    /// Batch size for training.
    /// </summary>
    int BatchSize { get; }

    /// <summary>
    /// Maximum number of examples to store in memory from previous tasks.
    /// </summary>
    int MemorySize { get; }

    /// <summary>
    /// Regularization strength for preventing forgetting (e.g., lambda in EWC).
    /// </summary>
    T RegularizationStrength { get; }

    /// <summary>
    /// Validates the configuration.
    /// </summary>
    /// <returns>True if the configuration is valid; otherwise, false.</returns>
    bool IsValid();
}
