using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for VFIT video frame interpolation transformer.
/// </summary>
/// <remarks>
/// <para>
/// VFIT (Shi et al., CVPR 2022) uses vision transformers for multi-frame interpolation:
/// - Multi-frame input: takes multiple input frames (typically 4: two before and two after the
///   target) to provide richer temporal context than 2-frame methods
/// - Temporal transformer: applies temporal self-attention across the multiple input frames,
///   learning long-range temporal dependencies and motion patterns that span multiple frames
/// - Spatial-temporal factorization: factorizes the full 3D attention into separate spatial
///   (within each frame) and temporal (across frames) attention for efficiency
/// - Progressive synthesis: generates the intermediate frame progressively from coarse to fine,
///   with transformer attention applied at each resolution level
/// </para>
/// <para>
/// <b>For Beginners:</b> VFIT uses more than just two frames to create the in-between frame.
/// By looking at a wider window of frames (2 before and 2 after), it can better understand
/// complex motions like acceleration, deceleration, and periodic movements.
/// </para>
/// </remarks>
public class VFITOptions : NeuralNetworkOptions
{
    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of input frames (typically 4).</summary>
    public int NumInputFrames { get; set; } = 4;

    /// <summary>Gets or sets the number of temporal transformer layers.</summary>
    public int NumTemporalLayers { get; set; } = 4;

    /// <summary>Gets or sets the number of spatial transformer layers.</summary>
    public int NumSpatialLayers { get; set; } = 4;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumHeads { get; set; } = 8;

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
