using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for ShiftNet channel-shifting video denoising.
/// </summary>
/// <remarks>
/// <para>
/// ShiftNet (Maggioni et al., 2021) uses feature-level temporal shifting for denoising:
/// - Channel shifting: shifts feature channels along the temporal dimension (some channels
///   come from past frames, some from future) without explicit alignment or flow, providing
///   zero-cost temporal information exchange
/// - Shift-and-aggregate: after shifting, local convolutions aggregate the temporally mixed
///   features, implicitly learning to handle motion without optical flow
/// - U-Net backbone: standard encoder-decoder with skip connections, where each conv block
///   incorporates channel shifting for temporal awareness
/// - Efficient design: no motion estimation or attention needed, making it faster than
///   flow-based or attention-based temporal methods
/// </para>
/// <para>
/// <b>For Beginners:</b> ShiftNet uses a clever trick to denoise video: instead of computing
/// expensive optical flow, it simply shifts some feature channels to come from past or future
/// frames. This free operation lets the network naturally learn to use temporal information
/// for denoising without any extra computation cost.
/// </para>
/// </remarks>
public class ShiftNetOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public ShiftNetOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ShiftNetOptions(ShiftNetOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumBlocks = other.NumBlocks;
        NumShifts = other.NumShifts;
        ShiftRadius = other.ShiftRadius;
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

    /// <summary>Gets or sets the number of encoder/decoder blocks.</summary>
    public int NumBlocks { get; set; } = 4;

    /// <summary>Gets or sets the number of temporal shifts per block.</summary>
    public int NumShifts { get; set; } = 4;

    /// <summary>Gets or sets the temporal radius for shifting.</summary>
    public int ShiftRadius { get; set; } = 2;

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
