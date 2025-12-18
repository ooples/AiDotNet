using AiDotNet.Data.Structures;
using AiDotNet.Models.Results;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for self-supervised loss functions used in meta-learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Self-supervised losses enable meta-learning without explicit labels by creating
/// supervisory signals from the data itself. This is particularly useful in
/// few-shot learning scenarios where labeled data is scarce.
/// </para>
/// <para><b>For Beginners:</b> Self-supervised learning is like learning from
/// examples without needing a teacher to tell you the right answers:
///
/// Instead of: "This image is a cat" (supervised)
/// The model learns: "These two patches are from the same image" (self-supervised)
///
/// Common techniques:
/// - Rotation prediction: Was the image rotated?
/// - Jigsaw puzzle: Can the model reorder shuffled patches?
/// - Contrastive learning: Which pairs are similar/different?
/// </para>
/// </remarks>
public interface ISelfSupervisedLoss<T, TInput, TOutput> : ILossFunction<T>
{
    /// <summary>
    /// Creates a self-supervised task from unlabeled data.
    /// </summary>
    /// <param name="unlabeledInput">Unlabeled input data.</param>
    /// <returns>A meta-learning task with self-supervised labels.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates learning tasks automatically
    /// from data without human labels:</para>
    ///
    /// Example for rotation prediction:
    /// 1. Take an image of a cat
    /// 2. Create 4 versions: 0°, 90°, 180°, 270° rotated
    /// 3. Label becomes: [0, 90, 180, 270] degrees
    /// 4. Model learns to predict rotation angle
    ///
    /// The learned features help with the actual task (e.g., identifying cats).
    /// </remarks>
    IMetaLearningTask<T, TInput, TOutput> CreateSelfSupervisedTask(TInput unlabeledInput);

    /// <summary>
    /// Gets the type of self-supervised task.
    /// </summary>
    /// <value>
    /// The specific self-supervised learning approach.
    /// </value>
    SelfSupervisedTaskType TaskType { get; }

    /// <summary>
    /// Applies data augmentation for self-supervised learning.
    /// </summary>
    /// <param name="input">Input data to augment.</param>
    /// <param name="augmentationParams">Parameters for augmentation.</param>
    /// <returns>Augmented input data.</returns>
    /// <remarks>
    /// Augmentations are essential for creating diverse self-supervised tasks.
    /// Common augmentations include rotation, cropping, color jitter, noise, etc.
    /// </remarks>
    TInput ApplyAugmentation(TInput input, Dictionary<string, object> augmentationParams);
}