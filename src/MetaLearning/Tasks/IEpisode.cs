using AiDotNet.LinearAlgebra;

namespace AiDotNet.MetaLearning.Tasks;

/// <summary>
/// Represents a single episode in meta-learning, containing support and query sets
/// for N-way K-shot learning tasks.
/// </summary>
/// <typeparam name="T">The numeric type (float, double)</typeparam>
/// <remarks>
/// For Beginners:
/// An episode is a mini-training task used in meta-learning. For example, in 5-way 1-shot learning:
/// - N (way) = 5: You have 5 different classes
/// - K (shot) = 1: You have 1 example per class in the support set
/// The support set is used for quick adaptation, and the query set is used for evaluation.
/// </remarks>
public interface IEpisode<T> where T : struct
{
    /// <summary>
    /// Gets the support set data (training examples for adaptation).
    /// Shape: [N*K, ...] where N is the number of classes and K is shots per class.
    /// </summary>
    Tensor<T> SupportData { get; }

    /// <summary>
    /// Gets the support set labels.
    /// Shape: [N*K] with values in range [0, N-1].
    /// </summary>
    Tensor<T> SupportLabels { get; }

    /// <summary>
    /// Gets the query set data (test examples for evaluation after adaptation).
    /// Shape: [N*Q, ...] where Q is the number of query examples per class.
    /// </summary>
    Tensor<T> QueryData { get; }

    /// <summary>
    /// Gets the query set labels.
    /// Shape: [N*Q] with values in range [0, N-1].
    /// </summary>
    Tensor<T> QueryLabels { get; }

    /// <summary>
    /// Gets the number of classes in this episode (N-way).
    /// </summary>
    int NumWays { get; }

    /// <summary>
    /// Gets the number of support examples per class (K-shot).
    /// </summary>
    int NumShots { get; }

    /// <summary>
    /// Gets the number of query examples per class.
    /// </summary>
    int NumQueries { get; }

    /// <summary>
    /// Gets the task identifier or name.
    /// </summary>
    string TaskId { get; }
}
