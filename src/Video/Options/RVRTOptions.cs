using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the RVRT recurrent video restoration transformer.
/// </summary>
/// <remarks>
/// <para>
/// RVRT (Liang et al., NeurIPS 2022) combines recurrent processing with transformers:
/// - Recurrent frame grouping: processes video in overlapping clips of ClipSize frames,
///   with hidden states propagated between clips for long-range temporal modeling
/// - Guided deformable attention (GDA): attention offsets are guided by optical flow,
///   combining the efficiency of deformable attention with explicit motion information
/// - Multi-scale temporal fusion: features from different temporal scales are fused
///   through a hierarchical structure
/// - Applicable to multiple tasks: SR, deblurring, and denoising in one architecture
/// </para>
/// <para>
/// <b>For Beginners:</b> RVRT processes video in small groups of frames at a time,
/// passing information forward like a memory. Within each group, it uses "guided
/// deformable attention" -- the model knows where to look in neighboring frames because
/// optical flow tells it where objects moved. This makes it both fast (small groups)
/// and accurate (guided by motion).
/// </para>
/// </remarks>
public class RVRTOptions : NeuralNetworkOptions
{
    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    /// <remarks>The paper uses 96 for small, 120 for base, 180 for large variant.</remarks>
    public int NumFeatures { get; set; } = 120;

    /// <summary>Gets or sets the number of transformer blocks per stage.</summary>
    public int NumBlocks { get; set; } = 4;

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the number of frames per recurrent clip.</summary>
    /// <remarks>Frames are grouped into clips of this size for recurrent processing.
    /// The paper uses 2 for small, 4 for base, 6 for large.</remarks>
    public int ClipSize { get; set; } = 4;

    /// <summary>Gets or sets the number of recurrent frame groups for temporal propagation.</summary>
    public int NumFrameGroups { get; set; } = 2;

    /// <summary>Gets or sets the number of attention heads for guided deformable attention.</summary>
    public int NumHeads { get; set; } = 6;

    /// <summary>Gets or sets the number of sampling points per deformable attention head.</summary>
    /// <remarks>More points increases accuracy but also computation cost.</remarks>
    public int NumSamplingPoints { get; set; } = 4;

    /// <summary>Gets or sets the spatial window size for local attention.</summary>
    public int WindowSize { get; set; } = 8;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 4e-4;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}
