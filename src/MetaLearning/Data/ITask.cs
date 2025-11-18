namespace AiDotNet.MetaLearning.Data;

/// <summary>
/// Represents a meta-learning task with support and query sets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<T>, Tensor<T>).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> In meta-learning, a task represents a single learning problem.
/// Think of it like a mini-dataset with two parts:
/// - Support Set: The "training" examples for this specific task (K examples per class)
/// - Query Set: The "test" examples to evaluate how well the model adapted to this task
///
/// For example, in a 5-way 1-shot image classification task:
/// - You have 5 classes (5-way)
/// - Each class has 1 example in the support set (1-shot)
/// - The query set has additional examples to test adaptation
///
/// This structure allows the model to learn how to quickly adapt to new tasks.
/// </para>
/// </remarks>
public interface ITask<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the support set input data (examples used for task adaptation).
    /// </summary>
    /// <value>The support set input data.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The support set is like the "few examples" you give to the model
    /// to help it learn a new task. In K-shot learning, this contains K examples per class.
    /// </para>
    /// </remarks>
    TInput SupportInput { get; }

    /// <summary>
    /// Gets the support set output labels.
    /// </summary>
    /// <value>The support set output labels.</value>
    TOutput SupportOutput { get; }

    /// <summary>
    /// Gets the query set input data (examples used for evaluation).
    /// </summary>
    /// <value>The query set input data.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The query set is like the "test examples" that check how well
    /// the model learned from the support set. These are not used during adaptation, only for evaluation.
    /// </para>
    /// </remarks>
    TInput QueryInput { get; }

    /// <summary>
    /// Gets the query set output labels.
    /// </summary>
    /// <value>The query set output labels.</value>
    TOutput QueryOutput { get; }

    /// <summary>
    /// Gets the number of classes (ways) in this task.
    /// </summary>
    /// <value>The number of classes.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In N-way K-shot learning, this is the N.
    /// For example, in 5-way 1-shot learning, NumWays = 5 (5 different classes to distinguish).
    /// </para>
    /// </remarks>
    int NumWays { get; }

    /// <summary>
    /// Gets the number of examples per class in the support set (shots).
    /// </summary>
    /// <value>The number of shots per class.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In N-way K-shot learning, this is the K.
    /// For example, in 5-way 1-shot learning, NumShots = 1 (only 1 example per class).
    /// </para>
    /// </remarks>
    int NumShots { get; }

    /// <summary>
    /// Gets the number of query examples per class.
    /// </summary>
    /// <value>The number of query examples per class.</value>
    int NumQueryPerClass { get; }

    /// <summary>
    /// Gets the task identifier or name.
    /// </summary>
    /// <value>The task identifier.</value>
    string TaskId { get; }
}
