using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for self-supervised loss functions used in meta-learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Self-supervised learning creates artificial tasks from unlabeled data, allowing models
/// to learn useful representations without explicit labels. This is particularly valuable
/// in meta-learning where the query set is often large but unlabeled.
/// </para>
/// <para><b>For Beginners:</b> Self-supervised learning is like learning by creating your own practice problems.
///
/// Example: Rotation prediction for images
/// - Take an unlabeled image
/// - Rotate it by 0°, 90°, 180°, or 270°
/// - Train the model to predict which rotation was applied
/// - The model learns spatial relationships and features without needing class labels
///
/// This is powerful because:
/// 1. You can use unlabeled data (which is often abundant)
/// 2. The model learns useful features automatically
/// 3. These features help with the actual task (classification, etc.)
///
/// Think of it like learning to recognize faces by first learning to identify if a photo is upside down.
/// You don't need to know who the person is to learn about facial features!
/// </para>
/// <para><b>Common Self-Supervised Tasks:</b>
/// - <b>Rotation Prediction:</b> Predict rotation angle (0°, 90°, 180°, 270°)
/// - <b>Jigsaw Puzzles:</b> Solve scrambled image patches
/// - <b>Colorization:</b> Predict color from grayscale
/// - <b>Context Prediction:</b> Predict spatial relationships between patches
/// - <b>Contrastive Learning:</b> Learn to distinguish similar vs dissimilar examples
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("SelfSupervisedLoss")]
public interface ISelfSupervisedLoss<T>
{
    /// <summary>
    /// Creates a self-supervised task from unlabeled input data.
    /// </summary>
    /// <param name="input">Unlabeled input data (e.g., images).</param>
    /// <returns>
    /// A tuple containing:
    /// - augmentedX: Transformed input data for the self-supervised task
    /// - augmentedY: Labels for the self-supervised task (e.g., rotation angles)
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method transforms unlabeled data into a supervised learning problem
    /// by creating artificial labels based on the transformation applied.
    /// </para>
    /// <para><b>For Beginners:</b> This converts "unlabeled data" into a "labeled learning problem".
    ///
    /// Example for rotation prediction:
    /// - Input: 10 unlabeled images
    /// - Output: 40 labeled images (each original rotated 4 times: 0°, 90°, 180°, 270°)
    /// - Labels: [0, 1, 2, 3] indicating which rotation was applied
    ///
    /// The model learns to recognize rotations, which teaches it about:
    /// - Edge orientations
    /// - Spatial relationships
    /// - Object structure
    ///
    /// These learned features are useful for the actual classification task!
    /// </para>
    /// </remarks>
    (TInput augmentedX, TOutput augmentedY) CreateTask<TInput, TOutput>(TInput input);
}
