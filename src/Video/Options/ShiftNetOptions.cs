using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for Shift-Net, the grouped spatial-temporal shift video restorer (Li et al., CVPR 2023),
/// in its denoising configuration.
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow the authors' release (github.com/dasongli1/Shift-Net, basicsr/models/archs/gshift_denoise1.py and
/// options/davis_denoise.yml): 24-channel full-resolution features refined by five U-Nets before and after the shift
/// encoder, an 80-channel shift encoder at half resolution, spatial shifts of 4 and 8 pixels, and Adam at 4e-4
/// (betas 0.9, 0.99) cosine-annealed to 1e-7 over 450k iterations on BasicSR's PSNR loss with weight 0.5.
/// </para>
/// <para>
/// <b>For Beginners:</b> Shift-Net cleans a video by sliding slices of each frame's features in time (to the
/// neighbouring frames) and in space (by a few pixels in many directions), letting the following convolutions pick
/// whichever slide lines up. Sliding is free, so it gets the benefit of alignment without computing motion.
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
    public ShiftNetOptions(ShiftNetOptions other) : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        NumFeatures = other.NumFeatures;
        RefinementFeatures = other.RefinementFeatures;
        UNetGrowth = other.UNetGrowth;
        NumUNets = other.NumUNets;
        ShiftStride = other.ShiftStride;
        NoiseLevel = other.NoiseLevel;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        MinLearningRate = other.MinLearningRate;
        CosineIterations = other.CosineIterations;
        PsnrLossWeight = other.PsnrLossWeight;
    }

    #region Architecture

    /// <summary>
    /// Gets or sets the shift encoder's channels (80); a multiple of 16, since half of them are split into the eight
    /// shift groups of the spatial shift.
    /// </summary>
    public int NumFeatures { get; set; } = 80;

    /// <summary>Gets or sets the full-resolution channels of the feature extractor and refinement U-Nets (24).</summary>
    public int RefinementFeatures { get; set; } = 24;

    /// <summary>Gets or sets the channels each refinement U-Net level adds when it halves the resolution (12).</summary>
    public int UNetGrowth { get; set; } = 12;

    /// <summary>Gets or sets the refinement U-Nets before the shift encoder, and again after it (5).</summary>
    public int NumUNets { get; set; } = 5;

    /// <summary>Gets or sets the spatial shift s: the shift groups move by s and 2s pixels (4).</summary>
    public int ShiftStride { get; set; } = 4;

    /// <summary>
    /// Gets or sets the noise standard deviation (in the frames' own units) that inference's noise map carries; null
    /// estimates it from the frames. Training uses the RMS of noisy minus clean. Default: 20/255.
    /// </summary>
    public double? NoiseLevel { get; set; } = 20.0 / 255.0;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets Adam's initial learning rate (4e-4).</summary>
    public double LearningRate { get; set; } = 4e-4;

    /// <summary>Gets or sets the learning rate the cosine schedule anneals to (1e-7).</summary>
    public double MinLearningRate { get; set; } = 1e-7;

    /// <summary>Gets or sets the iterations of the cosine schedule (450000).</summary>
    public int CosineIterations { get; set; } = 450000;

    /// <summary>Gets or sets the weight of BasicSR's PSNR loss (0.5).</summary>
    public double PsnrLossWeight { get; set; } = 0.5;

    #endregion
}
