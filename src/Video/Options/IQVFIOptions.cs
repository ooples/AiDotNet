using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for IQ-VFI image quality-aware video frame interpolation.
/// </summary>
/// <remarks>
/// <para>
/// IQ-VFI (2024) adapts interpolation based on input image quality:
/// - Quality assessment module: estimates per-pixel quality scores for input frames using a
///   learned no-reference image quality assessment (NR-IQA) branch, identifying regions with
///   noise, blur, compression artifacts, or other degradations
/// - Degradation-adaptive flow: the optical flow estimation network receives quality maps as
///   additional conditioning, so it can be more conservative in degraded regions (where flow
///   estimation is unreliable) and more aggressive in clean regions
/// - Quality-guided fusion: the blending weights between warped frames incorporate quality
///   scores, favoring the higher-quality frame contribution in each spatial region
/// - Quality-aware training: training uses a quality-stratified sampling strategy that ensures
///   the model sees diverse degradation levels and learns robust interpolation for each
/// </para>
/// <para>
/// <b>For Beginners:</b> Most frame interpolation methods assume input frames are clean and
/// high-quality. IQ-VFI first checks how good each part of the input frames is, then adjusts
/// its interpolation strategy accordingly. This means it works better on real-world videos
/// that may have noise, blur, or compression artifacts.
/// </para>
/// </remarks>
public class IQVFIOptions : NeuralNetworkOptions
{
    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of quality assessment blocks.</summary>
    /// <remarks>Controls the depth of the quality estimation sub-network.</remarks>
    public int NumQualityBlocks { get; set; } = 3;

    /// <summary>Gets or sets the number of flow refinement iterations.</summary>
    public int NumFlowRefinementIters { get; set; } = 4;

    /// <summary>Gets or sets the number of pyramid levels for multi-scale processing.</summary>
    public int NumPyramidLevels { get; set; } = 4;

    /// <summary>Gets or sets the quality threshold below which conservative interpolation is used.</summary>
    /// <remarks>Pixels with quality scores below this threshold use more conservative flow estimates.</remarks>
    public double QualityThreshold { get; set; } = 0.3;

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
