using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the StableVideoSR temporal-conditioned diffusion video super-resolution model.
/// </summary>
/// <remarks>
/// <para>
/// Options for the Upscale-A-Video compatibility entry point. The implementation uses
/// a temporal U-Net, temporal VAE decoder, low-resolution noise conditioning, overlapping
/// temporal windows, DDIM sampling, and flow-guided recurrent latent propagation.
/// </para>
/// <para>
/// <b>For Beginners:</b> StableVideoSR takes the popular Stable Diffusion image AI and
/// extends it to handle video. It adds special "temporal" modules that look at neighboring
/// frames to ensure the output video is smooth and flicker-free, not just a sequence
/// of independently upscaled images.
/// </para>
/// </remarks>
public class StableVideoSROptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public StableVideoSROptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public StableVideoSROptions(StableVideoSROptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumDenoisingSteps = other.NumDenoisingSteps;
        NumTemporalModules = other.NumTemporalModules;
        ScaleFactor = other.ScaleFactor;
        GuidanceScale = other.GuidanceScale;
        LatentDim = other.LatentDim;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
        WarmupSteps = other.WarmupSteps;
        LatentScaleFactor = other.LatentScaleFactor;
        MaximumNoiseLevel = other.MaximumNoiseLevel;
        TemporalWindowSize = other.TemporalWindowSize;
        TemporalWindowOverlap = other.TemporalWindowOverlap;
        EnableFlowGuidedPropagation = other.EnableFlowGuidedPropagation;
        NoiseLevel = other.NoiseLevel;
        PropagationSteps = other.PropagationSteps.ToArray();
        Prompt = other.Prompt;
        NegativePrompt = other.NegativePrompt;
    }

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of UNet feature channels.</summary>
    public int NumFeatures { get; set; } = 256;

    /// <summary>Gets or sets the number of denoising steps.</summary>
    public int NumDenoisingSteps { get; set; } = 75;

    /// <summary>Gets or sets the released count of stage-level temporal modules.</summary>
    public int NumTemporalModules { get; set; } = 9;

    /// <summary>Compatibility alias for the pre-fidelity option name.</summary>
    [Obsolete("Use NumTemporalModules. Upscale-A-Video has nine stage modules, not two attention layers per block.")]
    public int NumTemporalLayers
    {
        get => NumTemporalModules;
        set => NumTemporalModules = value;
    }

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the classifier-free guidance scale.</summary>
    public double GuidanceScale { get; set; } = 9.0;

    /// <summary>Gets or sets the latent space dimension.</summary>
    public int LatentDim { get; set; } = 4;

    /// <summary>Stable Diffusion x4-upscaler VAE latent scale.</summary>
    public double LatentScaleFactor { get; set; } = 0.08333;

    /// <summary>Maximum low-resolution noise conditioning level.</summary>
    public int MaximumNoiseLevel { get; set; } = 350;

    /// <summary>Number of frames in each overlapping temporal denoising window.</summary>
    public int TemporalWindowSize { get; set; } = 8;

    /// <summary>Number of frames blended between adjacent temporal windows.</summary>
    public int TemporalWindowOverlap { get; set; } = 2;

    /// <summary>Enables the paper's bidirectional flow-guided recurrent latent propagation.</summary>
    public bool EnableFlowGuidedPropagation { get; set; } = true;

    /// <summary>Low-resolution degradation noise level passed to the 1,000-entry class embedding.</summary>
    public int NoiseLevel { get; set; } = 20;

    /// <summary>
    /// Zero-based denoising iteration indices at which flow-guided x0 propagation runs.
    /// The released command-line default is empty; propagation requires bidirectional RAFT flows.
    /// </summary>
    public int[] PropagationSteps { get; set; } = [];

    /// <summary>Default text guidance prompt.</summary>
    public string Prompt { get; set; } = "best quality, extremely detailed";

    /// <summary>Released command-line negative prompt used for classifier-free guidance.</summary>
    public string NegativePrompt { get; set; } = "blur, worst quality";

    /// <summary>Rejects settings that the native paper graph cannot honor.</summary>
    public void ValidateNativePaperContract()
    {
        if (NumFeatures != 256)
            throw new ArgumentOutOfRangeException(nameof(NumFeatures), "Upscale-A-Video uses 256 base channels.");
        if (NumTemporalModules != 9)
            throw new ArgumentOutOfRangeException(nameof(NumTemporalModules),
                "Upscale-A-Video uses four down-stage, one mid-stage, and four up-stage temporal modules.");
        if (ScaleFactor != 4 || LatentDim != 4)
            throw new ArgumentException("Upscale-A-Video requires 4x output and four latent channels.");
        if (Math.Abs(LatentScaleFactor - 0.08333) > 1e-10)
            throw new ArgumentOutOfRangeException(nameof(LatentScaleFactor), "The x4-upscaler VAE scale is 0.08333.");
        if (MaximumNoiseLevel != 350 || NoiseLevel < 0 || NoiseLevel > MaximumNoiseLevel)
            throw new ArgumentOutOfRangeException(nameof(NoiseLevel), "The degradation noise level must be in [0,350].");
        if (NumDenoisingSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(NumDenoisingSteps));
        if (GuidanceScale < 0 || double.IsNaN(GuidanceScale))
            throw new ArgumentOutOfRangeException(nameof(GuidanceScale));
        if (TemporalWindowSize <= 0 || TemporalWindowOverlap < 0 || TemporalWindowOverlap >= TemporalWindowSize)
            throw new ArgumentException("Temporal overlap must be non-negative and smaller than the window.");
        if (PropagationSteps.Any(step => step < 0 || step >= NumDenoisingSteps) ||
            PropagationSteps.Distinct().Count() != PropagationSteps.Length)
            throw new ArgumentException("Propagation steps must be unique denoising indices in range.", nameof(PropagationSteps));
    }

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 5e-5;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>Gets or sets the warmup steps.</summary>
    public int WarmupSteps { get; set; } = 1000;

    #endregion
}
