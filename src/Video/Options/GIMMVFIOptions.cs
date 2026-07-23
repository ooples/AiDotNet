using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the GIMM-VFI generalizable implicit motion modeling.
/// </summary>
/// <remarks>
/// <para>
/// GIMM-VFI (NeurIPS 2024) uses implicit neural representations for continuous-time motion:
/// - Implicit motion function: learns a continuous function M(x, y, t) that maps any spatial
///   position (x, y) and any timestep t in [0, 1] to a motion vector, enabling interpolation
///   at arbitrary (non-uniform) time intervals without retraining
/// - Motion encoding network: encodes the two input frames into a shared motion latent space
///   using cross-correlation features, which the implicit function queries to produce per-pixel
///   motion at any desired timestep
/// - Generalizable across timesteps: a single forward pass of the motion encoder produces a
///   representation that the implicit function can query at any t, unlike methods that need
///   separate inference per timestep
/// - Adaptive sampling: the implicit function can be queried at higher density in regions with
///   complex motion and lower density in static regions for efficient computation
/// </para>
/// <para>
/// <b>For Beginners:</b> GIMM-VFI learns a smooth, continuous "motion field" that describes
/// how everything in the scene moves over time. Once it processes two frames, it can generate
/// a new frame at ANY point in time between them (not just the midpoint). This is great for
/// creating variable slow-motion effects or non-uniform frame rate conversion.
/// </para>
/// </remarks>
public class GIMMVFIOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public GIMMVFIOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GIMMVFIOptions(GIMMVFIOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumEncoderBlocks = other.NumEncoderBlocks;
        ImplicitDim = other.ImplicitDim;
        NumImplicitLayers = other.NumImplicitLayers;
        NumFrequencies = other.NumFrequencies;
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

    /// <summary>Gets or sets the number of motion encoder residual blocks.</summary>
    public int NumEncoderBlocks { get; set; } = 6;

    /// <summary>Gets or sets the dimension of the implicit motion representation.</summary>
    /// <remarks>Higher dimension captures more complex motion patterns.</remarks>
    public int ImplicitDim { get; set; } = 256;

    /// <summary>Gets or sets the number of MLP layers in the implicit function.</summary>
    public int NumImplicitLayers { get; set; } = 4;

    /// <summary>Gets or sets the number of positional encoding frequencies.</summary>
    /// <remarks>Higher values capture finer spatial/temporal detail.</remarks>
    public int NumFrequencies { get; set; } = 6;

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
