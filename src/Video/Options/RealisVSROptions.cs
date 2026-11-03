using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Options for <see cref="Enhancement.RealisVSR{T}"/>: Wan 2.1-1.3B with the Consistency-Preserved ControlNet, trained with the
/// High-Frequency Rectified Diffusion Loss (Zhao et al. 2025).
/// </summary>
/// <remarks>
/// <para>
/// The paper names its backbone (Wan-1.3B, whose released configuration supplies the DiT and VAE defaults here), the loss
/// weights, the HOG bins and the optimizer. It does not state the ControlNet's depth, the HOG cell and block sizes, the
/// sampler or the training timestep distribution, and the authors' repository is no longer published; those defaults are
/// documented on each property.
/// </para>
/// <para><b>For Beginners:</b> the defaults build the paper's 1.6B-parameter model. Shrink <see cref="ModelDim"/>,
/// <see cref="NumLayers"/>, <see cref="ControlNetLayers"/> and the VAE widths for experiments on modest hardware.</para>
/// </remarks>
public class RealisVSROptions : NeuralNetworkOptions
{
    /// <summary>Creates the paper's configuration.</summary>
    public RealisVSROptions()
    {
    }

    /// <summary>Copies <paramref name="other"/>.</summary>
    public RealisVSROptions(RealisVSROptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        ModelDim = other.ModelDim;
        FfnDim = other.FfnDim;
        NumHeads = other.NumHeads;
        NumLayers = other.NumLayers;
        FrequencyDim = other.FrequencyDim;
        TextDim = other.TextDim;
        TextLength = other.TextLength;
        NormEpsilon = other.NormEpsilon;
        ControlNetLayers = other.ControlNetLayers;
        ConditionScale = other.ConditionScale;
        VaeDim = other.VaeDim;
        VaeDimMultipliers = (int[])other.VaeDimMultipliers.Clone();
        VaeResidualBlocks = other.VaeResidualBlocks;
        ScaleFactor = other.ScaleFactor;
        NumFrames = other.NumFrames;
        SamplingSteps = other.SamplingSteps;
        SigmaShift = other.SigmaShift;
        NumTrainTimesteps = other.NumTrainTimesteps;
        WaveletWeights = (double[])other.WaveletWeights.Clone();
        HogBins = other.HogBins;
        HogCellSize = other.HogCellSize;
        HogBlockSize = other.HogBlockSize;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
    }

    #region Wan 2.1-1.3B

    /// <summary>The DiT width (Wan-1.3B: 1536).</summary>
    public int ModelDim { get; set; } = 1536;

    /// <summary>The feed-forward width (Wan-1.3B: 8960).</summary>
    public int FfnDim { get; set; } = 8960;

    /// <summary>The attention heads (Wan-1.3B: 12).</summary>
    public int NumHeads { get; set; } = 12;

    /// <summary>L_main, the main DiT's blocks (Wan-1.3B: 30).</summary>
    public int NumLayers { get; set; } = 30;

    /// <summary>The sinusoidal time embedding's width (Wan-1.3B: 256).</summary>
    public int FrequencyDim { get; set; } = 256;

    /// <summary>The umT5 embedding width the DiT reads (4096).</summary>
    public int TextDim { get; set; } = 4096;

    /// <summary>The prompt's token count (512).</summary>
    public int TextLength { get; set; } = 512;

    /// <summary>The norms' ε (1e-6).</summary>
    public double NormEpsilon { get; set; } = 1e-6;

    /// <summary>The Wan VAE's base width (96).</summary>
    public int VaeDim { get; set; } = 96;

    /// <summary>The Wan VAE's stage multipliers ({1, 2, 4, 4}: 8x in space, 4x in time).</summary>
    public int[] VaeDimMultipliers { get; set; } = { 1, 2, 4, 4 };

    /// <summary>The Wan VAE's residual blocks per stage (2).</summary>
    public int VaeResidualBlocks { get; set; } = 2;

    #endregion

    #region Consistency-Preserved ControlNet

    /// <summary>L_CPC, the ControlNet branch's blocks.</summary>
    /// <remarks>
    /// Not stated in the paper. Six is derived from its parameter count: 1.6B in all against Wan-1.3B leaves about 0.26B for
    /// the branch, six Wan-1.3B blocks; six also divides the 30 main blocks, so r = 5 and every branch feature is used.
    /// </remarks>
    public int ControlNetLayers { get; set; } = 6;

    /// <summary>γ's initial value, the paper's condition scale (1.0).</summary>
    public double ConditionScale { get; set; } = 1.0;

    #endregion

    #region Video and sampling

    /// <summary>The upscaling factor (4).</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>The frames of a training clip (49).</summary>
    public int NumFrames { get; set; } = 49;

    /// <summary>The Euler steps of the rectified-flow sampler.</summary>
    /// <remarks>Not stated in the paper; Wan 2.1's own default (50) is used.</remarks>
    public int SamplingSteps { get; set; } = 50;

    /// <summary>The flow-matching shift s, σ' = s·σ / (1 + (s − 1)·σ).</summary>
    /// <remarks>Not stated in the paper; Wan 2.1's text-to-video default (5.0) is used for training and sampling.</remarks>
    public double SigmaShift { get; set; } = 5.0;

    /// <summary>The training schedule's length; training draws σ uniformly from its shifted values (1000).</summary>
    public int NumTrainTimesteps { get; set; } = 1000;

    #endregion

    #region High-Frequency Rectified Diffusion Loss

    /// <summary>The Haar band weights {LL, LH, HL, HH} of the wavelet loss (Eq. 4: {1, 2, 2, 2}).</summary>
    public double[] WaveletWeights { get; set; } = { 1.0, 2.0, 2.0, 2.0 };

    /// <summary>The HOG loss's orientation bins over [0°, 180°] (Eq. 5: 9).</summary>
    public int HogBins { get; set; } = 9;

    /// <summary>The HOG cell side in latent pixels.</summary>
    /// <remarks>Not stated in the paper; scikit-image's default (8) is used.</remarks>
    public int HogCellSize { get; set; } = 8;

    /// <summary>The HOG block side in cells.</summary>
    /// <remarks>Not stated in the paper; scikit-image's default (3) is used.</remarks>
    public int HogBlockSize { get; set; } = 3;

    #endregion

    #region Model Loading

    /// <summary>The ONNX model's path in inference mode.</summary>
    public string? ModelPath { get; set; }

    /// <summary>The ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>AdamW's learning rate (Sec. 4.1: 1e-5).</summary>
    public double LearningRate { get; set; } = 1e-5;

    /// <summary>AdamW's weight decay (Sec. 4.1: 1e-4).</summary>
    public double WeightDecay { get; set; } = 1e-4;

    #endregion
}
