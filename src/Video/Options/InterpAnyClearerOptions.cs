using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for InterpAnyClearer plug-in module for clearer anytime interpolation.
/// </summary>
/// <remarks>
/// <para>
/// InterpAnyClearer (Zheng et al., ECCV 2024 Oral) resolves velocity ambiguity in VFI:
/// - Velocity-ambiguity analysis: identifies that standard VFI models produce blurry results
///   when motion speed varies within a scene, because a single flow vector per pixel cannot
///   represent multiple plausible velocities simultaneously
/// - Plug-in velocity predictor: a lightweight auxiliary network that predicts per-pixel velocity
///   magnitude from the input frame pair, conditioning the base VFI model to select the correct
///   motion hypothesis for each region
/// - Multi-velocity training: during training, the model sees multiple velocity annotations per
///   pixel (from different temporal distances), learning to disambiguate fast vs slow motion
/// - Base-model agnostic: designed as a plug-in that wraps any existing VFI model (RIFE, IFRNet,
///   AMT, EMA-VFI, etc.) without modifying its architecture, only adding velocity conditioning
/// </para>
/// <para>
/// <b>For Beginners:</b> When objects in a video move at different speeds, standard interpolation
/// can get confused and produce blurry results. InterpAnyClearer adds a small "speed detector"
/// that tells the main model how fast each part of the image is moving, so it can produce
/// sharp results even when some objects move fast and others are still.
/// </para>
/// </remarks>
public class InterpAnyClearerOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public InterpAnyClearerOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public InterpAnyClearerOptions(InterpAnyClearerOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumVelocityBlocks = other.NumVelocityBlocks;
        NumVelocityBins = other.NumVelocityBins;
        NumPyramidLevels = other.NumPyramidLevels;
        UseVelocityGuidedWarping = other.UseVelocityGuidedWarping;
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

    /// <summary>Gets or sets the number of velocity predictor blocks.</summary>
    /// <remarks>Controls the depth of the auxiliary velocity estimation network.</remarks>
    public int NumVelocityBlocks { get; set; } = 4;

    /// <summary>Gets or sets the number of velocity bins for discretized speed estimation.</summary>
    /// <remarks>Higher values provide finer velocity discrimination but need more training data.</remarks>
    public int NumVelocityBins { get; set; } = 16;

    /// <summary>Gets or sets the number of pyramid levels for multi-scale velocity estimation.</summary>
    public int NumPyramidLevels { get; set; } = 3;

    /// <summary>Gets or sets whether to use velocity-guided warping.</summary>
    /// <remarks>When true, the warping operation is conditioned on predicted velocity magnitude.</remarks>
    public bool UseVelocityGuidedWarping { get; set; } = true;

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
