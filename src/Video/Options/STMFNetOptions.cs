using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for STMFNet spatio-temporal multi-flow network.
/// </summary>
/// <remarks>
/// <para>
/// STMFNet (2022) uses multiple optical flows in spatio-temporal space:
/// - Multi-flow estimation: estimates multiple (typically 4) optical flow fields between the
///   input frames, each capturing different motion hypotheses for ambiguous regions like
///   occlusion boundaries, transparent objects, and repeating textures
/// - Spatio-temporal feature volume: constructs a 4D (height x width x time x channel) feature
///   volume from the input frames and all estimated flow fields, capturing the full motion
///   context in a unified representation
/// - Flow selection network: a learned network that selects the best flow hypothesis for each
///   pixel by comparing warped features from each flow field, choosing the one that produces
///   the most consistent result
/// - Residual refinement: after flow-based warping, a refinement network corrects remaining
///   artifacts using the multi-flow feature volume as context
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of guessing one "best" motion for each pixel, STMFNet makes
/// multiple guesses (flows) and then picks the best one for each part of the image. This works
/// much better in tricky areas like where objects overlap or where motion is ambiguous.
/// </para>
/// </remarks>
public class STMFNetOptions : NeuralNetworkOptions
{
    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of flow hypotheses per pixel.</summary>
    public int NumFlowHypotheses { get; set; } = 4;

    /// <summary>Gets or sets the number of spatio-temporal fusion blocks.</summary>
    public int NumFusionBlocks { get; set; } = 4;

    /// <summary>Gets or sets the number of refinement blocks.</summary>
    public int NumRefineBlocks { get; set; } = 2;

    /// <summary>Gets or sets the number of pyramid levels for multi-scale flow.</summary>
    public int NumPyramidLevels { get; set; } = 3;

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
