using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the EMA-VFI swin-based inter-frame attention model.
/// </summary>
/// <remarks>
/// <para>
/// EMA-VFI (Zhang et al., CVPR 2023) extracts motion and appearance via inter-frame attention:
/// - Swin-based cross-attention: shifted window cross-attention between frame pairs to extract
///   motion correspondence without explicit optical flow computation
/// - Dual-branch extraction: one branch captures motion dynamics (displacement features)
///   while the other captures appearance information (texture, color) from both frames
/// - Bilateral motion estimation: bidirectional motion fields estimated simultaneously
///   using cross-attention scores as soft correspondence weights
/// - Multi-scale feature fusion: hierarchical feature pyramid with cross-scale connections
///   for handling both small and large motions
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of explicitly computing optical flow (how pixels move),
/// EMA-VFI uses attention to simultaneously figure out "what moved where" (motion) and
/// "what does it look like" (appearance). By processing both together, it avoids errors
/// from bad flow estimates and produces cleaner interpolated frames.
/// </para>
/// </remarks>
public class EMAVFIOptions : NeuralNetworkOptions
{
    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels in the encoder.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of swin cross-attention blocks per scale.</summary>
    /// <remarks>The paper uses 4 blocks per scale in the base configuration.</remarks>
    public int NumSwinBlocks { get; set; } = 4;

    /// <summary>Gets or sets the number of attention heads in each swin block.</summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>Gets or sets the swin window size for local attention.</summary>
    public int WindowSize { get; set; } = 8;

    /// <summary>Gets or sets the number of pyramid scales for multi-scale fusion.</summary>
    /// <remarks>Each scale halves spatial resolution. 3 scales = 1x, 1/2x, 1/4x.</remarks>
    public int NumScales { get; set; } = 3;

    /// <summary>Gets or sets whether to use bidirectional motion estimation.</summary>
    public bool BidirectionalMotion { get; set; } = true;

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
