namespace AiDotNet.Compression;

/// <summary>
/// Provides default values for various compression parameters.
/// </summary>
/// <remarks>
/// <para>
/// This class defines recommended default values for compression parameters
/// that generally work well across a variety of models and tasks. These values
/// serve as starting points and can be fine-tuned for specific models.
/// </para>
/// <para><b>For Beginners:</b> These are sensible starting values for compressing models.
/// 
/// Instead of guessing what values to use for compression settings, this class provides
/// reasonable defaults that work well for most cases. You can adjust them later if needed
/// based on your specific requirements for model size and accuracy.
/// </para>
/// </remarks>
public static class CompressionDefaults
{
    /// <summary>
    /// Default bit precision for quantization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// 8-bit quantization offers a good balance between compression ratio and accuracy preservation.
    /// </para>
    /// <para><b>For Beginners:</b> 8-bit precision means each number takes 8 bits of storage.
    /// 
    /// This is 4x smaller than the standard 32-bit floating-point numbers,
    /// while usually maintaining good accuracy for most neural network models.
    /// </para>
    /// </remarks>
    public const int DefaultQuantizationPrecision = 8;

    /// <summary>
    /// Default sparsity target for pruning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A target of 0.7 means aiming to have 70% of the weights pruned (set to zero).
    /// </para>
    /// <para><b>For Beginners:</b> This means removing 70% of the least important connections.
    /// 
    /// Research has shown that many neural networks can maintain most of their accuracy
    /// even when 70% of their connections are removed, if done strategically.
    /// </para>
    /// </remarks>
    public const double DefaultPruningSparsity = 0.7;

    /// <summary>
    /// Default size ratio for the student model in knowledge distillation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A value of 0.5 indicates that the student model will have approximately half
    /// the capacity (e.g., hidden units, layers) of the teacher model.
    /// </para>
    /// <para><b>For Beginners:</b> The student model will be about half the size of the original.
    /// 
    /// This ratio has been found to work well for many applications, allowing significant
    /// compression while still providing enough capacity for the student to learn effectively.
    /// </para>
    /// </remarks>
    public const double DefaultDistillationStudentSize = 0.5;

    /// <summary>
    /// Default temperature parameter for knowledge distillation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The temperature parameter controls how "soft" the teacher model's probability
    /// distribution is when training the student model.
    /// </para>
    /// <para><b>For Beginners:</b> Temperature of 2.0 makes the teacher's outputs smoother.
    /// 
    /// A higher temperature:
    /// - Makes probability differences less extreme
    /// - Helps the student learn more from the relationships between classes
    /// - Often leads to better distillation performance
    /// </para>
    /// </remarks>
    public const double DefaultDistillationTemperature = 2.0;

    /// <summary>
    /// Default factor for low-rank approximation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A value of 0.25 means reducing the rank to approximately 25% of the original.
    /// </para>
    /// <para><b>For Beginners:</b> This compresses matrices to 25% of their original complexity.
    /// 
    /// For example, if a layer has a 1000×1000 weight matrix with an effective rank of 1000,
    /// applying this factor would approximate it with matrices that have an effective rank of 250.
    /// </para>
    /// </remarks>
    public const double DefaultLowRankFactor = 0.25;

    /// <summary>
    /// Default number of clusters for weight clustering.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Using 256 clusters allows representing each weight with a single byte index.
    /// </para>
    /// <para><b>For Beginners:</b> This groups all weights into 256 different values.
    /// 
    /// Instead of storing thousands or millions of unique weight values, the weights are
    /// grouped into 256 clusters. Each weight is then stored as an index (0-255) into
    /// a table of these 256 representative values.
    /// </para>
    /// </remarks>
    public const int DefaultClusterCount = 256;

    /// <summary>
    /// Default acceptable accuracy loss (in percentage points).
    /// </summary>
    /// <remarks>
    /// <para>
    /// A value of 0.02 means accepting a maximum accuracy decrease of 2 percentage points.
    /// </para>
    /// <para><b>For Beginners:</b> This means we accept losing at most 2% accuracy.
    /// 
    /// For example, if the original model had 95% accuracy, we would accept the compressed
    /// model having no less than 93% accuracy.
    /// </para>
    /// </remarks>
    public const double DefaultAcceptableAccuracyLoss = 0.02;
}