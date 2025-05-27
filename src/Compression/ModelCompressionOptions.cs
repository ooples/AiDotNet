namespace AiDotNet.Compression;

using AiDotNet.Enums;
using AiDotNet.Models.Options;

/// <summary>
/// Defines options for model compression techniques.
/// </summary>
/// <remarks>
/// <para>
/// This class encapsulates various parameters and settings for compressing machine learning models.
/// Model compression reduces model size and potentially improves inference speed while
/// attempting to maintain model accuracy within acceptable thresholds.
/// </para>
/// <para><b>For Beginners:</b> Model compression makes AI models smaller and faster.
/// 
/// Think of model compression like compressing a large file:
/// - You can make the model take up less storage space
/// - The model can run faster, especially on mobile devices or edge devices
/// - There's usually a small trade-off in accuracy, but often it's barely noticeable
/// 
/// This is especially important when deploying models to production environments with
/// limited resources, such as mobile devices, embedded systems, or web browsers.
/// </para>
/// </remarks>
public class ModelCompressionOptions
{
    /// <summary>
    /// Gets or sets the logging options for the compression process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Configure how detailed logging should be during the compression process.
    /// Logs can help with debugging and understanding the compression process.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much information is recorded during compression.
    /// 
    /// You can configure:
    /// - Whether logging is enabled
    /// - Where log files are stored
    /// - How detailed the logs should be
    /// - Whether to also log to the console
    /// 
    /// Logging is particularly useful when debugging compression issues or
    /// when you want to analyze the compression process in detail.
    /// </para>
    /// </remarks>
    public LoggingOptions LoggingOptions { get; set; } = new LoggingOptions();
    
    /// <summary>
    /// Gets or sets a more detailed description of the compression technique.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Provides additional details about the specific compression approach being used.
    /// This can be useful for documentation and reporting purposes.
    /// </para>
    /// <para><b>For Beginners:</b> This lets you add a human-readable description of your compression.
    /// 
    /// While the Technique property specifies the general category (e.g., "Quantization"),
    /// this property lets you provide more specific details like:
    /// - "8-bit weight quantization with clustering"
    /// - "Progressive pruning with fine-tuning"
    /// 
    /// This information is useful when comparing different compression methods
    /// or documenting your compression pipeline.
    /// </para>
    /// </remarks>
    public string CompressionTechniqueNote { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the compression technique to use.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Different compression techniques offer different trade-offs between model size,
    /// inference speed, and accuracy preservation.
    /// </para>
    /// <para><b>For Beginners:</b> This determines the general approach to make your model smaller.
    /// Each technique has its own strengths:
    /// - Quantization: Reduces numerical precision (like using 8 bits instead of 32)
    /// - Pruning: Removes unimportant connections in the model
    /// - Knowledge Distillation: Creates a smaller model that mimics a larger one
    /// - Low Rank Factorization: Compresses matrices that store model weights
    /// - Huffman Coding: Applies efficient binary encoding to model parameters
    /// </para>
    /// </remarks>
    public CompressionTechnique Technique { get; set; } = CompressionTechnique.Quantization;

    /// <summary>
    /// Gets or sets the target compression ratio.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the ratio of the original model size to the target compressed size.
    /// A value of 4.0 means the target is to make the model 1/4 of its original size.
    /// </para>
    /// <para><b>For Beginners:</b> This specifies how much smaller you want the model to be.
    /// 
    /// For example:
    /// - A value of 2.0 means you want the model to be half its original size
    /// - A value of 4.0 means you want the model to be quarter its original size
    /// - Higher values mean more compression but potentially more accuracy loss
    /// </para>
    /// </remarks>
    public double TargetCompressionRatio { get; set; } = 4.0;

    /// <summary>
    /// Gets or sets the maximum acceptable loss in accuracy.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents the maximum acceptable decrease in model accuracy after compression.
    /// A value of 0.02 means the compressed model should maintain accuracy within 2% of the original.
    /// </para>
    /// <para><b>For Beginners:</b> This sets a limit on how much accuracy you're willing to sacrifice.
    /// 
    /// For example:
    /// - A value of 0.01 means you'll accept at most a 1% drop in accuracy
    /// - If the original model was 95% accurate, you'll accept no less than 94%
    /// - The compression process will try to respect this limit
    /// </para>
    /// </remarks>
    public double MaxAcceptableAccuracyLoss { get; set; } = 0.02;

    /// <summary>
    /// Gets or sets the precision for quantization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For quantization-based compression, this specifies the bit width to use for
    /// representing model parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how many bits are used to store each number.
    /// 
    /// Neural networks typically use 32-bit or 16-bit numbers for calculations, but:
    /// - 8-bit numbers are much smaller (75% smaller than 32-bit)
    /// - 4-bit numbers are even smaller (87.5% smaller than 32-bit)
    /// - Lower precision means smaller size but potentially less accuracy
    /// </para>
    /// </remarks>
    public int QuantizationPrecision { get; set; } =
        CompressionDefaults.DefaultQuantizationPrecision;

    /// <summary>
    /// Gets or sets the sparsity target for pruning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For pruning-based compression, this specifies the target percentage of parameters
    /// to prune (set to zero).
    /// </para>
    /// <para><b>For Beginners:</b> This determines what percentage of connections to remove.
    /// 
    /// Neural networks have many connections, but not all are equally important:
    /// - A value of 0.5 means removing 50% of the least important connections
    /// - A value of 0.8 means removing 80% of the least important connections
    /// - Higher values mean smaller models but potentially less accuracy
    /// </para>
    /// </remarks>
    public double PruningSparsityTarget { get; set; } = 
        CompressionDefaults.DefaultPruningSparsity;

    /// <summary>
    /// Gets or sets the size of the student model for knowledge distillation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For knowledge distillation, this specifies the size of the student model
    /// relative to the teacher model.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how big the simplified "student" model will be.
    /// 
    /// Knowledge distillation works by teaching a smaller model to mimic a larger one:
    /// - A value of 0.5 means the student model will be half the size of the original
    /// - A value of 0.25 means the student model will be quarter the size
    /// - Smaller values mean more compression but potentially more accuracy loss
    /// </para>
    /// </remarks>
    public double DistillationStudentSize { get; set; } = 
        CompressionDefaults.DefaultDistillationStudentSize;

    /// <summary>
    /// Gets or sets the temperature for knowledge distillation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For knowledge distillation, this parameter controls the softness of the teacher
    /// model's probability distribution that the student learns from.
    /// </para>
    /// <para><b>For Beginners:</b> This affects how the student model learns from the teacher.
    /// 
    /// Temperature in knowledge distillation:
    /// - Higher values (>1) make the teacher's output distribution smoother
    /// - This helps the student learn more about the relationships between classes
    /// - Typical values range from 1 to 10
    /// </para>
    /// </remarks>
    public double DistillationTemperature { get; set; } = 
        CompressionDefaults.DefaultDistillationTemperature;

    /// <summary>
    /// Gets or sets the rank reduction factor for low-rank factorization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For low-rank factorization, this specifies how much to reduce the rank of the
    /// weight matrices.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much to simplify the model's matrix math.
    /// 
    /// Low-rank factorization works by approximating large matrices with smaller ones:
    /// - A value of 0.25 means reducing the rank to 25% of the original
    /// - This can significantly reduce model size and computation
    /// - Lower values mean more compression but potentially less accuracy
    /// </para>
    /// </remarks>
    public double LowRankFactor { get; set; } = 
        CompressionDefaults.DefaultLowRankFactor;
        
    /// <summary>
    /// Gets or sets whether to apply mixed precision.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When enabled, different layers or parameter types may use different precisions
    /// based on their sensitivity to quantization.
    /// </para>
    /// <para><b>For Beginners:</b> This lets the compression use different precision for different parts.
    /// 
    /// Instead of using the same precision everywhere:
    /// - Important parts can use higher precision (more accurate)
    /// - Less important parts can use lower precision (more compression)
    /// - This gives a better balance between size and accuracy
    /// </para>
    /// </remarks>
    public bool UseMixedPrecision { get; set; } = false;
        
    /// <summary>
    /// Gets or sets whether the compression is dynamic (determined during inference).
    /// </summary>
    /// <remarks>
    /// <para>
    /// When enabled, some compression decisions are made dynamically during inference
    /// rather than statically during the compression process.
    /// </para>
    /// <para><b>For Beginners:</b> This determines if compression adjusts during model use.
    /// 
    /// Static compression (false):
    /// - Compression is applied once when preparing the model
    /// - The model stays the same during all future use
    /// 
    /// Dynamic compression (true):
    /// - Parts of compression happen while the model runs
    /// - Can adapt to different inputs for better performance
    /// - Usually more complex but can be more efficient
    /// </para>
    /// </remarks>
    public bool IsDynamicCompression { get; set; } = false;
        
    /// <summary>
    /// Gets or sets whether to verify the compressed model's accuracy.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When enabled, the compression process will verify that the compressed model
    /// maintains accuracy within the specified MaxAcceptableAccuracyLoss threshold.
    /// </para>
    /// <para><b>For Beginners:</b> This determines if the system checks accuracy after compression.
    /// 
    /// When enabled:
    /// - After compression, the model is tested against a validation set
    /// - If the accuracy drop is too high, compression settings are adjusted
    /// - Provides safety but requires validation data and more processing time
    /// </para>
    /// </remarks>
    public bool VerifyAccuracy { get; set; } = true;
}