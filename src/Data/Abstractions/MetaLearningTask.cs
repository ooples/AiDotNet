using AiDotNet.LinearAlgebra;

namespace AiDotNet.Data.Abstractions;

/// <summary>
/// Represents a single meta-learning task for few-shot learning, containing support and query sets.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <remarks>
/// <para>
/// In meta-learning, particularly few-shot learning, a task is a small classification or regression problem
/// sampled from a larger dataset. Each task contains a support set (for adapting/learning) and a query set
/// (for evaluating the adaptation).
/// </para>
/// <para><b>For Beginners:</b> Meta-learning is about "learning to learn" - training a model to quickly adapt
/// to new tasks with only a few examples. Think of it like learning a language:
/// - Traditional learning: Learn one specific language from thousands of examples
/// - Meta-learning: Learn the general skill of language acquisition so you can learn new languages faster
///
/// A MetaLearningTask represents one mini-problem in this process:
/// - <b>Support Set:</b> A few labeled examples the model can study (like a mini training set)
/// - <b>Query Set:</b> Examples to test how well the model adapted (like a mini test set)
///
/// For example, in 5-way 3-shot classification:
/// - The support set has 5 classes with 3 examples each (15 total examples)
/// - The query set has examples from those same 5 classes to test performance
///
/// The model learns from many such tasks, developing the ability to quickly adapt to new tasks.
/// </para>
/// <para>
/// <b>Thread Safety:</b> This class is not thread-safe. Create separate instances for concurrent access.
/// </para>
/// <para>
/// <b>Performance:</b> This is a lightweight container class with O(1) property access.
/// Memory usage depends on the size of the tensors stored.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a 5-way 3-shot task
/// // Support set: 5 classes × 3 shots = 15 examples
/// // Query set: 5 classes × 10 queries = 50 examples
///
/// var supportX = new Tensor&lt;double&gt;(new[] { 15, 784 }); // 15 images, 784 pixels each
/// var supportY = new Tensor&lt;double&gt;(new[] { 15, 5 });   // 15 one-hot labels, 5 classes
/// var queryX = new Tensor&lt;double&gt;(new[] { 50, 784 });   // 50 query images
/// var queryY = new Tensor&lt;double&gt;(new[] { 50, 5 });     // 50 query labels
///
/// var task = new MetaLearningTask&lt;double&gt;
/// {
///     SupportSetX = supportX,
///     SupportSetY = supportY,
///     QuerySetX = queryX,
///     QuerySetY = queryY
/// };
///
/// // Use with MAML, Reptile, or SEAL algorithms
/// var innerLoss = model.Train(task.SupportSetX, task.SupportSetY);
/// var outerLoss = model.Evaluate(task.QuerySetX, task.QuerySetY);
/// </code>
/// </example>
public class MetaLearningTask<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the input features for the support set.
    /// </summary>
    /// <value>
    /// Input data containing the examples used for task adaptation.
    /// Shape depends on TInput type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;, double[]).
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> These are the input examples the model can study to adapt to this task.
    /// Think of them as the "practice problems" before a quiz. In a 5-way 3-shot image classification task:
    /// - You have 5 different categories (ways)
    /// - With 3 example images each (shots)
    /// - Total: 15 example images to learn from
    ///
    /// The model looks at these examples to quickly understand what makes each class unique.
    /// </para>
    /// </remarks>
    public TInput SupportSetX { get; set; }

    /// <summary>
    /// Gets or sets the target labels for the support set.
    /// </summary>
    /// <value>
    /// Output data containing the labels corresponding to each example in the support set.
    /// Shape depends on TOutput type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;, double[]).
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> These are the correct answers for each example in the support set.
    /// They tell the model which category each example belongs to.
    ///
    /// For example, in animal classification:
    /// - If SupportSetX contains images of cats, dogs, and birds
    /// - SupportSetY tells which images are cats (label 0), dogs (label 1), or birds (label 2)
    ///
    /// Labels can be:
    /// - Class indices: [0, 0, 0, 1, 1, 1, 2, 2, 2, ...] for 3-shot
    /// - One-hot encoded: [[1,0,0], [1,0,0], [1,0,0], [0,1,0], ...] for 3-shot
    /// </para>
    /// </remarks>
    public TOutput SupportSetY { get; set; }

    /// <summary>
    /// Gets or sets the input features for the query set.
    /// </summary>
    /// <value>
    /// Input data containing the examples used for evaluating task adaptation.
    /// Shape depends on TInput type, typically larger than the support set size.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> These are the examples used to test how well the model adapted to this task.
    /// Think of them as the "quiz questions" after studying the practice problems (support set).
    ///
    /// Continuing the 5-way example:
    /// - After studying the 3 examples per class (support set)
    /// - The model is tested on new, unseen examples from those same classes
    /// - Typically 10-15 queries per class (50-75 total examples)
    ///
    /// The model must correctly classify these new examples based only on what it learned
    /// from the few support examples. This tests the model's ability to quickly adapt.
    /// </para>
    /// </remarks>
    public TInput QuerySetX { get; set; }

    /// <summary>
    /// Gets or sets the target labels for the query set.
    /// </summary>
    /// <value>
    /// Output data containing the true labels for evaluating predictions on the query set.
    /// Shape depends on TOutput type.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> These are the correct answers for the query set examples.
    /// They're used to calculate how accurately the model can classify new examples after
    /// adapting to the task using only the support set.
    ///
    /// The query accuracy is the key metric in few-shot learning:
    /// - High query accuracy = Model successfully adapted from few examples
    /// - Low query accuracy = Model struggled to generalize from limited data
    ///
    /// During meta-training, the model learns to maximize query set performance across
    /// many different tasks, developing strong few-shot learning abilities.
    /// </para>
    /// </remarks>
    public TOutput QuerySetY { get; set; }
}
