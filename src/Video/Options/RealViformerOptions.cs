using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the RealViformer video informer for real-world video SR.
/// </summary>
/// <remarks>
/// <para>
/// RealViformer (Zhang and Yao, ECCV 2024) investigates attention for real-world VSR:
/// - Channel attention (CA): SE-style channel attention that recalibrates feature channels
///   based on global statistics, found more effective than spatial attention for real-world
///   degradations with complex noise patterns
/// - Temporal propagation: bidirectional recurrent feature propagation with channel
///   attention at each step for adaptive temporal fusion
/// - Informer-style sparse attention: ProbSparse self-attention that selects only the
///   top-k most informative queries, reducing quadratic complexity for longer sequences
/// - Real-world degradation handling: trained with second-order degradation modeling
///   (blur, noise, resize, JPEG) for practical video restoration
/// </para>
/// <para>
/// <b>For Beginners:</b> RealViformer is designed for real-world video (phone recordings,
/// compressed streams), not just lab-quality test videos. It found that paying attention
/// to "which color channels matter" (channel attention) works better than "which spatial
/// locations matter" for handling the messy, complex degradations in real footage. It also
/// uses a trick from time-series forecasting (Informer) to efficiently handle longer videos.
/// </para>
/// </remarks>
public class RealViformerOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public RealViformerOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public RealViformerOptions(RealViformerOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumResBlocks = other.NumResBlocks;
        ScaleFactor = other.ScaleFactor;
        ChannelReductionRatio = other.ChannelReductionRatio;
        SparseTopKFactor = other.SparseTopKFactor;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
    }

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of residual channel attention blocks.</summary>
    public int NumResBlocks { get; set; } = 20;

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the channel attention reduction ratio.</summary>
    /// <remarks>The squeeze ratio in SE-style channel attention (features / ratio).</remarks>
    public int ChannelReductionRatio { get; set; } = 16;

    /// <summary>Gets or sets the ProbSparse attention top-k sampling factor.</summary>
    /// <remarks>Controls sparsity: ln(L) * factor queries are selected from L total.</remarks>
    public int SparseTopKFactor { get; set; } = 5;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 2e-4;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}
