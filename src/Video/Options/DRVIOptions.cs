using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the DRVI disentangled representations model.
/// </summary>
/// <remarks>
/// <para>
/// DRVI (2024) disentangles content and motion for video interpolation:
/// - Disentangled encoders: separate encoders for content (appearance, texture, color) and
///   motion (displacement, deformation) that process frame pairs independently
/// - Content encoder: extracts appearance features invariant to motion, shared across all
///   timesteps so the model doesn't re-extract appearance at each interpolation point
/// - Motion encoder: captures inter-frame displacement fields at multiple scales, enabling
///   the model to handle both global camera motion and local object motion
/// - Disentangled decoder: recombines content and motion representations with learned gating
///   at each scale, allowing fine control over which content features are warped by which
///   motion components
/// </para>
/// <para>
/// <b>For Beginners:</b> DRVI separates "what things look like" from "how things move".
/// By processing appearance and motion independently, it can better handle cases where
/// objects look similar but move differently, or where the same object appears in different
/// lighting conditions across frames.
/// </para>
/// </remarks>
public class DRVIOptions : NeuralNetworkOptions
{
    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of content encoder blocks.</summary>
    public int NumContentBlocks { get; set; } = 4;

    /// <summary>Gets or sets the number of motion encoder blocks.</summary>
    public int NumMotionBlocks { get; set; } = 4;

    /// <summary>Gets or sets the number of decoder blocks for recombination.</summary>
    public int NumDecoderBlocks { get; set; } = 4;

    /// <summary>Gets or sets the number of pyramid scales.</summary>
    public int NumScales { get; set; } = 3;

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
