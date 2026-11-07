using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for <see cref="AiDotNet.Video.FrameInterpolation.FLAVR{T}"/>, flow-agnostic video
/// representations for fast frame interpolation (Kalluri et al., WACV 2023).
/// </summary>
/// <remarks>
/// <para>
/// The defaults are the release's (tarun005/FLAVR: model/FLAVR_arch.py, model/resnet_3D.py, config.py, main.py): the
/// unet_18 encoder (R3D-18 with feature gating, no batch norm), four input frames and one output frame for 2x
/// interpolation; <see cref="Deep"/> returns the unet_34 encoder. Setting <see cref="NumOutputFrames"/> to 3 or 7 gives
/// the release's 4x and 8x models.
/// </para>
/// <para>
/// <b>For Beginners:</b> FLAVR never estimates motion. It stacks four frames into a short clip, runs 3D (space and
/// time) convolutions over it in a U-Net, and directly paints the in-between frame or frames.
/// </para>
/// </remarks>
public class FLAVROptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public FLAVROptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public FLAVROptions(FLAVROptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        BaseChannels = other.BaseChannels;
        LayerBlocks = (int[])other.LayerBlocks.Clone();
        NumInputFrames = other.NumInputFrames;
        NumOutputFrames = other.NumOutputFrames;
        LearningRate = other.LearningRate;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        PlateauFactor = other.PlateauFactor;
        PlateauPatience = other.PlateauPatience;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

    /// <summary>The release's unet_34 encoder (R3D-34 block counts).</summary>
    public static FLAVROptions Deep() => new() { LayerBlocks = new[] { 3, 4, 6, 3 } };

    #region Architecture

    /// <summary>Gets or sets the encoder's first width; the four stages are 1, 2, 4 and 8 times it (64).</summary>
    public int BaseChannels { get; set; } = 64;

    /// <summary>Gets or sets the residual blocks in each of the encoder's four stages (unet_18: 2, 2, 2, 2).</summary>
    public int[] LayerBlocks { get; set; } = { 2, 2, 2, 2 };

    /// <summary>Gets or sets the context frames, half before and half after the gap (4).</summary>
    public int NumInputFrames { get; set; } = 4;

    /// <summary>Gets or sets the evenly spaced frames synthesized in the gap (1 for 2x; 3 for 4x; 7 for 8x).</summary>
    public int NumOutputFrames { get; set; } = 1;

    #endregion

    #region Training

    /// <summary>Gets or sets Adam's initial learning rate (2e-4).</summary>
    public double LearningRate { get; set; } = 2e-4;

    /// <summary>Gets or sets Adam's first-moment decay (0.9).</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>Gets or sets Adam's second-moment decay (0.99).</summary>
    public double Beta2 { get; set; } = 0.99;

    /// <summary>Gets or sets the factor the rate is multiplied by when the validation loss plateaus (0.5).</summary>
    public double PlateauFactor { get; set; } = 0.5;

    /// <summary>Gets or sets the epochs without improvement before the rate drops (5).</summary>
    public int PlateauPatience { get; set; } = 5;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion
}
