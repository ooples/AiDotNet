namespace AiDotNet.Data.Structures;

/// <summary>
/// Enumeration of self-supervised task types for meta-learning.
/// </summary>
/// <remarks>
/// <para>
/// Self-supervised tasks create learning signals without human labels by
/// leveraging the structure inherent in the data itself. These are
/// particularly valuable in few-shot and meta-learning scenarios where
/// labeled data is scarce.
/// </para>
/// <para><b>For Beginners:</b> These are different ways computers can learn
/// from pictures without humans telling them what's in the pictures:</para>
///
/// <b>Common Self-Supervised Tasks:</b>
/// - <b>Rotation:</b> "Was this image rotated?" (teaches about object orientation)
/// - <b>Jigsaw:</b> "Can you solve this picture puzzle?" (teaches about spatial relationships)
/// - <b>Contrastive:</b> "Which of these images are most similar?" (teaches about features)
/// - <b>Masking:</b> "What's missing from this picture?" (teaches about context)
/// </remarks>
public enum SelfSupervisedTaskType
{
    /// <summary>
    /// No self-supervised task (standard supervised learning).
    /// </summary>
    None,

    /// <summary>
    /// Rotation prediction: Model predicts the rotation angle applied to an image.
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// 1. Take an image
    /// 2. Rotate it by 0°, 90°, 180°, or 270°
    /// 3. Model learns to predict which rotation was applied
    /// 4. Forces model to learn about object orientation and features
    ///
    /// <para><b>Why it works:</b> To predict rotation, the model must understand
    /// what objects are and their typical orientation in the world.</para>
    /// </remarks>
    RotationPrediction,

    /// <summary>
    /// Jigsaw puzzle: Model learns to reorder shuffled image patches.
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// 1. Cut an image into a grid (e.g., 3x3)
    /// 2. Shuffle the pieces randomly
    /// 3. Model learns to put them back in correct order
    /// 4. Forces model to understand spatial relationships
    ///
    /// <para><b>Why it works:</b> Solving puzzles requires understanding object
    /// structure and continuity between patches.</para>
    /// </remarks>
    JigsawPuzzle,

    /// <summary>
    /// Contrastive learning: Model learns to distinguish similar from dissimilar pairs.
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// 1. Create pairs of images
    /// 2. Positive pairs: Two views/augmentations of same image
    /// 3. Negative pairs: Different images
    /// 4. Model learns to push positive pairs together, negative pairs apart
    ///
    /// <para><b>Why it works:</b> To distinguish similar from dissimilar, the model
    /// must learn meaningful features.</para>
    /// </remarks>
    ContrastiveLearning,

    /// <summary>
    /// Masked prediction: Model predicts missing parts of the input.
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// 1. Randomly mask/hide parts of the input
    /// 2. Model learns to predict what was masked
    /// 3. Can be applied to images (predict pixels), text (predict words), etc.
    ///
    /// <para><b>Why it works:</b> To predict missing parts, the model must understand
    /// context and patterns in the data.</para>
    /// </remarks>
    MaskedPrediction,

    /// <summary>
    /// Colorization: Model learns to predict colors for grayscale images.
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// 1. Convert color images to grayscale
    /// 2. Model learns to predict original colors
    /// 3. Forces understanding of object properties and scene context
    ///
    /// <para><b>Why it works:</b> Objects have characteristic colors (grass is green,
    /// sky is blue), so colorization requires semantic understanding.</para>
    /// </remarks>
    Colorization,

    /// <summary>
    /// Future prediction: Model predicts future frames in a sequence.
    /// </summary>
    /// <remarks>
    /// <para><b>How it works:</b></para>
    /// 1. Give model a sequence of frames (video, time series)
    /// 2. Model learns to predict next frame(s)
    /// 3. Forces understanding of dynamics and temporal patterns
    ///
    /// <para><b>Why it works:</b> Predicting the future requires understanding the
    /// underlying patterns and physics of the sequence.</para>
    /// </remarks>
    FuturePrediction,

    /// <summary>
    /// Custom user-defined self-supervised task.
    /// </summary>
    Custom
}