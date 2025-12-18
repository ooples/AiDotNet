namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a single meta-learning task with support and query sets for few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// IMetaLearningTask represents a single few-shot learning problem that a meta-learning
/// algorithm must solve. Each task contains a support set for rapid adaptation and a
/// query set for evaluation. This is the fundamental unit in meta-learning and
/// few-shot learning scenarios.
/// </para>
/// <para><b>For Beginners:</b> An IMetaLearningTask is like a mini-learning problem:</para>
///
/// Example: 5-way 3-shot image classification
/// - Support set: 5 classes × 3 examples = 15 labeled images to learn from
/// - Query set: 20-30 images from those same classes to test learning
///
/// The meta-learner trains on many such IMetaLearningTasks to learn how to quickly adapt
/// to new tasks with very few examples, developing general learning-to-learn capabilities.
/// </para>
/// </remarks>
public interface IMetaLearningTask<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the input features for the support set.
    /// </summary>
    /// <value>
    /// Input data containing examples for task adaptation.
    /// Shape depends on TInput type.
    /// </value>
    /// <remarks>
    /// The support set provides the labeled examples that the model can use
    /// to adapt to this specific task. In few-shot learning, this is typically
    /// a very small number of examples (1-5 per class).
    /// </remarks>
    TInput SupportInput { get; }

    /// <summary>
    /// Gets the target labels for the support set.
    /// </summary>
    /// <value>
    /// Output data containing labels corresponding to SupportInput.
    /// </value>
    /// <remarks>
    /// Labels can be class indices, one-hot encoded vectors, or other
    /// appropriate representations depending on the task type.
    /// </remarks>
    TOutput SupportOutput { get; }

    /// <summary>
    /// Gets the input features for the query set.
    /// </summary>
    /// <value>
    /// Input data for evaluating adaptation performance.
    /// Typically larger than the support set.
    /// </value>
    /// <remarks>
    /// The query set is used to test how well the model has adapted
    /// to the task using only the support set. The model is evaluated
    /// on the query set without seeing its labels.
    /// </remarks>
    TInput QueryInput { get; }

    /// <summary>
    /// Gets the target labels for the query set.
    /// </summary>
    /// <value>
    /// True labels for evaluating query set performance.
    /// </value>
    /// <remarks>
    /// These labels are used only for evaluation and loss computation.
    /// The model does not see them during adaptation.
    /// </remarks>
    TOutput QueryOutput { get; }

    /// <summary>
    /// Gets an optional name or identifier for the task.
    /// </summary>
    /// <value>
    /// Human-readable task name or null if not specified.
    /// </value>
    string? Name { get; }

    /// <summary>
    /// Gets additional metadata about the task.
    /// </summary>
    /// <value>
    /// Dictionary containing task-specific information.
    /// </value>
    /// <remarks>
    /// Can include information like:
    /// - Task difficulty
    /// - Data source
    /// - Task category
    /// - Number of classes
    /// - Other relevant attributes
    /// </remarks>
    Dictionary<string, object>? Metadata { get; }
}