using System.Collections.Generic;

namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a single meta-learning task for few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
[AiDotNet.Configuration.YamlConfigurable("MetaLearningTask")]
public interface IMetaLearningTask<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the input features for the support set.
    /// </summary>
    TInput SupportInput { get; }

    /// <summary>
    /// Gets the target labels for the support set.
    /// </summary>
    TOutput SupportOutput { get; }

    /// <summary>
    /// Gets the input features for the query set.
    /// </summary>
    TInput QueryInput { get; }

    /// <summary>
    /// Gets the target labels for the query set.
    /// </summary>
    TOutput QueryOutput { get; }

    /// <summary>
    /// Gets an optional name or identifier for the task.
    /// </summary>
    string? Name { get; }

    /// <summary>
    /// Gets the additional metadata about the task.
    /// </summary>
    Dictionary<string, object>? Metadata { get; }

    /// <summary>
    /// Gets the number of ways (classes) in this task.
    /// </summary>
    /// <remarks>
    /// In N-way K-shot learning, this represents the N (number of classes per task).
    /// For example, in 5-way 1-shot learning, NumWays = 5.
    /// </remarks>
    int NumWays { get; }

    /// <summary>
    /// Gets the number of shots (examples per class) in the support set.
    /// </summary>
    /// <remarks>
    /// In N-way K-shot learning, this represents the K (number of examples per class).
    /// For example, in 5-way 1-shot learning, NumShots = 1.
    /// </remarks>
    int NumShots { get; }

    /// <summary>
    /// Gets the number of query examples per class.
    /// </summary>
    /// <remarks>
    /// The number of examples in the query set for each class.
    /// Used for evaluating performance after adaptation.
    /// </remarks>
    int NumQueryPerClass { get; }

    // Aliases for compatibility - implementations should return corresponding property
    /// <summary>
    /// Gets the input features for the query set (alias for QueryInput).
    /// </summary>
    TInput QuerySetX { get; }

    /// <summary>
    /// Gets the target labels for the query set (alias for QueryOutput).
    /// </summary>
    TOutput QuerySetY { get; }

    /// <summary>
    /// Gets the input features for the support set (alias for SupportInput).
    /// </summary>
    TInput SupportSetX { get; }

    /// <summary>
    /// Gets the target labels for the support set (alias for SupportOutput).
    /// </summary>
    TOutput SupportSetY { get; }

    /// <summary>
    /// Gets or sets an optional identifier for the task.
    /// </summary>
    int? TaskId { get; set; }
}
