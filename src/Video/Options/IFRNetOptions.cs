using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for <see cref="AiDotNet.Video.FrameInterpolation.IFRNet{T}"/>, the intermediate feature refine
/// network for efficient frame interpolation (Kong et al., CVPR 2022).
/// </summary>
/// <remarks>
/// <para>
/// The defaults are IFRNet's (ltkong218/IFRNet, models/IFRNet.py); <see cref="Large"/> and <see cref="Small"/> return
/// IFRNet-L's and IFRNet-S's. Training follows train_vimeo90k.py.
/// </para>
/// <para>
/// <b>For Beginners:</b> IFRNet encodes both frames into a four-level pyramid, then decodes from the coarsest level up,
/// each decoder refining the two motions from the in-between frame and a feature of the in-between frame itself; the
/// last one also predicts how to blend the two warped frames and a correction.
/// </para>
/// </remarks>
public class IFRNetOptions : ModelOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public IFRNetOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public IFRNetOptions(IFRNetOptions other) : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderChannels = (int[])other.EncoderChannels.Clone();
        SideChannels = other.SideChannels;
        FirstKernelSize = other.FirstKernelSize;
        LearningRate = other.LearningRate;
        MinLearningRate = other.MinLearningRate;
        TotalIterations = other.TotalIterations;
        WeightDecay = other.WeightDecay;
        GeometryWeight = other.GeometryWeight;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

    /// <summary>IFRNet-L (models/IFRNet_L.py): wider, with a 7x7 first convolution.</summary>
    public static IFRNetOptions Large() => new()
    {
        EncoderChannels = new[] { 64, 96, 144, 192 },
        SideChannels = 64,
        FirstKernelSize = 7,
    };

    /// <summary>IFRNet-S (models/IFRNet_S.py).</summary>
    public static IFRNetOptions Small() => new()
    {
        EncoderChannels = new[] { 24, 36, 54, 72 },
        SideChannels = 24,
    };

    #region Architecture

    /// <summary>Gets or sets the encoder pyramid's four widths, 1/2 to 1/16 size (32, 48, 72, 96).</summary>
    public int[] EncoderChannels { get; set; } = { 32, 48, 72, 96 };

    /// <summary>Gets or sets the decoders' ResBlock side channels (32).</summary>
    public int SideChannels { get; set; } = 32;

    /// <summary>Gets or sets the encoder's first convolution kernel (3; IFRNet-L uses 7).</summary>
    public int FirstKernelSize { get; set; } = 3;

    #endregion

    #region Training

    /// <summary>Gets or sets AdamW's initial learning rate (1e-4).</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the rate the cosine schedule ends at (1e-5).</summary>
    public double MinLearningRate { get; set; } = 1e-5;

    /// <summary>Gets or sets the cosine schedule's length: 300 epochs of Vimeo90K's 51,312 triplets at batch 24.</summary>
    public int TotalIterations { get; set; } = 300 * (51312 / 24);

    /// <summary>Gets or sets AdamW's weight decay (0).</summary>
    public double WeightDecay { get; set; }

    /// <summary>Gets or sets the weight of the geometry consistency loss on the refined intermediate features (0.01).</summary>
    public double GeometryWeight { get; set; } = 0.01;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion
}
