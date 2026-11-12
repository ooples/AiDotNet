namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for ProDiff (Huang et al. 2022): FastSpeech 2's encoder and variance adaptor with a generator-based
/// diffusion denoiser, distilled from a 4-step teacher into a 2-step model.</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's (§4, §6.1, Table 4) with the values it leaves to its reference implementation
/// (Rongjiehuang/ProDiff, <c>modules/ProDiff/config</c>, <c>egs/egs_bases/tts</c>): phoneme embedding 256, a 3-layer
/// encoder pre-net, 4 FFT blocks (2 heads, filter 1024, kernels 9 and 1, dropout 0.05) and a final linear projection;
/// variance predictors kernel 3, filter 256, dropout 0.5; pitch lookup table 300; a denoiser of 20 residual layers,
/// 256 channels, kernel 3, no dilation; a cosine noise schedule; a 4-step teacher distilled into 2 steps; variance
/// losses weighted 0.1; Adam with β = (0.9, 0.98), ε = 1e-9 on the inverse-square-root schedule (factor 2, 2000 warmup
/// steps) and gradient-norm clip 1; 22.05 kHz, 1024-point STFT, hop 256, 80 mel bins.
/// </para>
/// <para><b>For Beginners:</b> These options configure the ProDiff model. Default values follow the original paper settings.</para>
/// </remarks>
public class ProDiffOptions : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ProDiffOptions(ProDiffOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        NumDiffusionSteps = other.NumDiffusionSteps;
        TeacherDiffusionSteps = other.TeacherDiffusionSteps;
        CosineScheduleOffset = other.CosineScheduleOffset;
        WarmupSteps = other.WarmupSteps;
        OptimizerBeta1 = other.OptimizerBeta1;
        OptimizerBeta2 = other.OptimizerBeta2;
        OptimizerEpsilon = other.OptimizerEpsilon;
        MaxGradientNorm = other.MaxGradientNorm;
        PrenetLayers = other.PrenetLayers;
        PrenetKernelSize = other.PrenetKernelSize;
        FftFilterSize = other.FftFilterSize;
        FftKernelSizes = (int[])other.FftKernelSizes.Clone();
        VariancePredictorFilterSize = other.VariancePredictorFilterSize;
        VariancePredictorKernelSize = other.VariancePredictorKernelSize;
        VariancePredictorDropout = other.VariancePredictorDropout;
        NumPitchBins = other.NumPitchBins;
        PitchMinHz = other.PitchMinHz;
        PitchMaxHz = other.PitchMaxHz;
        NumEnergyBins = other.NumEnergyBins;
        DenoiserLayers = other.DenoiserLayers;
        DenoiserChannels = other.DenoiserChannels;
        DenoiserDilationCycle = other.DenoiserDilationCycle;
        VarianceLossWeight = other.VarianceLossWeight;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Creates the paper's configuration.</summary>
    public ProDiffOptions()
    {
        EncoderDim = 256;
        DecoderDim = 256;
        HiddenDim = 256;
        NumEncoderLayers = 4;
        NumDecoderLayers = 0;
        NumHeads = 2;
        DropoutRate = 0.05;
        SampleRate = 22050;
        HopSize = 256;
        FftSize = 1024;
        MelChannels = 80;
        UsePostnet = false;
        // The inverse-square-root schedule's factor (reference lr 2.0 with rsqrt: 2 · min(s/w, 1) · max(w, s)^-0.5 · d^-0.5).
        LearningRate = 2.0;
        WeightDecay = 0.0;
    }

    /// <summary>Gets or sets the distilled model's diffusion steps T₂ (2).</summary>
    public int NumDiffusionSteps { get; set; } = 2;

    /// <summary>Gets or sets the teacher's diffusion steps T₁ (4); a multiple of <see cref="NumDiffusionSteps"/>.</summary>
    public int TeacherDiffusionSteps { get; set; } = 4;

    /// <summary>Gets or sets the cosine schedule's offset s (0.008, Nichol and Dhariwal 2021).</summary>
    public double CosineScheduleOffset { get; set; } = 0.008;

    /// <summary>Gets or sets the inverse-square-root schedule's warmup steps (2000).</summary>
    public int WarmupSteps { get; set; } = 2000;

    /// <summary>Gets or sets Adam's β₁ (0.9).</summary>
    public double OptimizerBeta1 { get; set; } = 0.9;

    /// <summary>Gets or sets Adam's β₂ (0.98).</summary>
    public double OptimizerBeta2 { get; set; } = 0.98;

    /// <summary>Gets or sets Adam's ε (1e-9, §6.1).</summary>
    public double OptimizerEpsilon { get; set; } = 1e-9;

    /// <summary>Gets or sets the global gradient-norm clip (1; 0 disables).</summary>
    public double MaxGradientNorm { get; set; } = 1.0;

    /// <summary>Gets or sets the encoder pre-net's convolutions (3, Table 4).</summary>
    public int PrenetLayers { get; set; } = 3;

    /// <summary>Gets or sets the encoder pre-net's kernel (5).</summary>
    public int PrenetKernelSize { get; set; } = 5;

    /// <summary>Gets or sets the FFT blocks' convolutional filter (1024).</summary>
    public int FftFilterSize { get; set; } = 1024;

    /// <summary>Gets or sets the FFT blocks' two convolution kernels (9, 1).</summary>
    public int[] FftKernelSizes { get; set; } = { 9, 1 };

    /// <summary>Gets or sets the variance predictors' filter (256).</summary>
    public int VariancePredictorFilterSize { get; set; } = 256;

    /// <summary>Gets or sets the variance predictors' kernel (3).</summary>
    public int VariancePredictorKernelSize { get; set; } = 3;

    /// <summary>Gets or sets the variance predictors' dropout (0.5).</summary>
    public double VariancePredictorDropout { get; set; } = 0.5;

    /// <summary>Gets or sets the pitch embedding's lookup-table size (300, §6.1).</summary>
    public int NumPitchBins { get; set; } = 300;

    /// <summary>Gets or sets the lowest quantized pitch in Hz (71, FastSpeech 2's range).</summary>
    public double PitchMinHz { get; set; } = 71.0;

    /// <summary>Gets or sets the highest quantized pitch in Hz (800).</summary>
    public double PitchMaxHz { get; set; } = 800.0;

    /// <summary>Gets or sets the energy embedding's bins (256).</summary>
    public int NumEnergyBins { get; set; } = 256;

    /// <summary>Gets or sets the denoiser's residual layers N (20).</summary>
    public int DenoiserLayers { get; set; } = 20;

    /// <summary>Gets or sets the denoiser's residual channels (256).</summary>
    public int DenoiserChannels { get; set; } = 256;

    /// <summary>Gets or sets the denoiser's dilation cycle (1: no dilation).</summary>
    public int DenoiserDilationCycle { get; set; } = 1;

    /// <summary>Gets or sets the weight on each variance loss (0.1, §4.5).</summary>
    public double VarianceLossWeight { get; set; } = 0.1;

    /// <summary>Gets or sets the seed of the training draws and the sampling noise, for repeatable runs.</summary>
    public int SamplingSeed { get; set; }
}
