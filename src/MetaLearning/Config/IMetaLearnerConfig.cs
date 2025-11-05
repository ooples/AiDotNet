namespace AiDotNet.MetaLearning.Config;

/// <summary>
/// Configuration interface for meta-learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric type (float, double)</typeparam>
public interface IMetaLearnerConfig<T> where T : struct
{
    /// <summary>
    /// Gets the inner loop learning rate (for task adaptation).
    /// </summary>
    T InnerLearningRate { get; }

    /// <summary>
    /// Gets the outer loop learning rate (for meta-optimization).
    /// </summary>
    T OuterLearningRate { get; }

    /// <summary>
    /// Gets the number of inner loop adaptation steps.
    /// </summary>
    int InnerSteps { get; }

    /// <summary>
    /// Gets the meta-batch size (number of tasks per meta-update).
    /// </summary>
    int MetaBatchSize { get; }

    /// <summary>
    /// Gets whether to use first-order approximation (faster but less accurate).
    /// </summary>
    bool FirstOrder { get; }

    /// <summary>
    /// Validates the configuration.
    /// </summary>
    /// <returns>True if valid, false otherwise</returns>
    bool IsValid();
}
