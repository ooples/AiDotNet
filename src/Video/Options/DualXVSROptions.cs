using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the DualX-VSR dual axial spatial-temporal transformer.
/// </summary>
/// <remarks>
/// <para>
/// DualX-VSR (2025) eliminates explicit motion compensation through dual axial attention:
/// - Dual axial attention: decomposes 3D (H x W x T) attention into two orthogonal axes
///   -- spatial-height-temporal and spatial-width-temporal -- reducing complexity from
///   O((HWT)^2) to O(HWT * max(H,W,T))
/// - Motion-free alignment: the dual axial attention implicitly captures inter-frame
///   correspondence without computing optical flow or deformable offsets
/// - Symmetric bidirectional processing: forward and backward temporal propagation with
///   shared axial attention weights
/// - Efficient design: linear complexity in the number of frames while maintaining
///   full spatial-temporal receptive field
/// </para>
/// <para>
/// <b>For Beginners:</b> Most video SR models need to figure out how objects moved between
/// frames (optical flow). DualX-VSR skips this step entirely by using a clever attention
/// pattern that looks along two crossing axes simultaneously, naturally capturing motion
/// without explicit computation. Think of it like looking at a crossword puzzle -- by
/// reading both across and down, you understand the full picture.
/// </para>
/// </remarks>
public class DualXVSROptions : NeuralNetworkOptions
{
    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of dual axial transformer blocks.</summary>
    /// <remarks>Each block contains both height-temporal and width-temporal axial attention.</remarks>
    public int NumAxialBlocks { get; set; } = 8;

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the number of attention heads for axial attention.</summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>Gets or sets the temporal window size for axial attention.</summary>
    /// <remarks>Limits temporal extent per attention operation for memory efficiency.</remarks>
    public int TemporalWindow { get; set; } = 7;

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
