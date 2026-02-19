using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for ToonCrafter cartoon/anime video frame interpolation.
/// </summary>
/// <remarks>
/// <para>
/// ToonCrafter (2024) specializes in cartoon and anime frame interpolation:
/// - Cartoon-aware diffusion: adapts a video diffusion model specifically for cartoon/anime
///   content, where motion is typically non-physical (e.g., smear frames, anticipation poses)
///   and cannot be captured by standard optical flow methods
/// - Style-preserving generation: uses CLIP-based style conditioning to maintain consistent
///   art style, line weight, and coloring throughout the interpolated sequence
/// - Large motion handling: the diffusion backbone can generate plausible in-betweens even
///   for extreme pose changes common in hand-drawn animation
/// - Sketch-guided control: optional sketch/line art conditioning for artist control over
///   intermediate poses and motion trajectories
/// </para>
/// <para>
/// <b>For Beginners:</b> Regular frame interpolation is designed for real-world videos and
/// fails on cartoons/anime because animated characters move in exaggerated, non-realistic
/// ways. ToonCrafter is specially trained on animation content to produce natural-looking
/// in-between frames for cartoons.
/// </para>
/// </remarks>
public class ToonCrafterOptions : NeuralNetworkOptions
{
    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of diffusion denoising steps.</summary>
    public int NumDiffusionSteps { get; set; } = 25;

    /// <summary>Gets or sets the number of U-Net residual blocks per level.</summary>
    public int NumResBlocks { get; set; } = 2;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>Gets or sets the guidance scale for classifier-free guidance.</summary>
    public double GuidanceScale { get; set; } = 7.5;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-5;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}
