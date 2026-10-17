using AiDotNet.Audio.Codecs;

namespace AiDotNet.Audio.Generation;

/// <summary>
/// Options for EnCodec ("High Fidelity Neural Audio Compression", Défossez et al. 2022). The defaults are the paper's
/// 24 kHz streamable model; <see cref="OfficialCheckpoint24kHz"/> and <see cref="OfficialCheckpoint48kHz"/> are the
/// configurations of the released checkpoints, which differ from the paper's text where noted.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> EnCodec turns audio into a short sequence of numbers (codes) and back. The number of
/// codebooks used per frame sets the bitrate: at 24 kHz each codebook adds 0.75 kbps, so 8 codebooks give 6 kbps.</para>
/// </remarks>
public class EnCodecOptions : NeuralAudioCodecOptions
{
    /// <summary>Creates the paper's 24 kHz streamable configuration.</summary>
    public EnCodecOptions()
    {
        SampleRate = 24000;
        Channels = 1;
        NumQuantizers = 32;
        CodebookSize = 1024;
        TargetBandwidthKbps = 6.0;
        SegmentSize = 24000;
    }

    // ---------------------------------------------------------------- encoder / decoder (§3.1)

    /// <summary>Gets or sets the base channel count C (32).</summary>
    public int Filters { get; set; } = 32;

    /// <summary>Gets or sets the strides in decoder order (8, 5, 4, 2); the encoder applies them reversed (2, 4, 5, 8),
    /// 320× at 24 kHz (75 frames per second).</summary>
    public int[] Ratios { get; set; } = [8, 5, 4, 2];

    /// <summary>Gets or sets the latent dimension D (128).</summary>
    public int Dimension { get; set; } = 128;

    /// <summary>Gets or sets the residual units per block (1).</summary>
    public int ResidualLayers { get; set; } = 1;

    /// <summary>Gets or sets the residual unit's two kernels: the paper's "two convolutions with kernel size 3" (3, 3);
    /// the released checkpoints use (3, 1).</summary>
    public int[] ResidualKernelSizes { get; set; } = [3, 3];

    /// <summary>Gets or sets the dilation growth over a block's residual units (2; reference).</summary>
    public int DilationBase { get; set; } = 2;

    /// <summary>Gets or sets the residual unit's channel reduction (2; reference).</summary>
    public int Compress { get; set; } = 2;

    /// <summary>Gets or sets whether the residual shortcut is the identity rather than a 1×1 convolution (false;
    /// reference).</summary>
    public bool TrueSkip { get; set; }

    /// <summary>Gets or sets the LSTM layers after the encoder's and before the decoder's convolutions (2).</summary>
    public int LstmLayers { get; set; } = 2;

    /// <summary>Gets or sets the first convolution's kernel (7).</summary>
    public int KernelSize { get; set; } = 7;

    /// <summary>Gets or sets the last convolution's kernel (7).</summary>
    public int LastKernelSize { get; set; } = 7;

    /// <summary>Gets or sets whether the model is streamable: causal padding and weight normalization (true at 24 kHz);
    /// otherwise padding is split evenly and each convolution is followed by a layer normalization over channels and
    /// time (§3.1, "Non-streamable").</summary>
    public bool Causal { get; set; } = true;

    /// <summary>Gets or sets whether convolution padding reflects the signal rather than zero-filling (true; reference).</summary>
    public bool ReflectPadding { get; set; } = true;

    // ---------------------------------------------------------------- non-streamable framing (§3.1)

    /// <summary>Gets or sets whether each chunk is divided by its RMS before encoding and multiplied back after decoding
    /// (false at 24 kHz; the 48 kHz model's "normalize each chunk").</summary>
    public bool NormalizeVolume { get; set; }

    /// <summary>Gets or sets the chunk length in seconds the encoder processes separately (null: the whole input; 1 s for
    /// the non-streamable 48 kHz model).</summary>
    public double? ChunkSeconds { get; set; }

    /// <summary>Gets or sets the overlap between chunks as a fraction of a chunk (0.01: "an overlap of 10 ms").</summary>
    public double ChunkOverlap { get; set; } = 0.01;

    // ---------------------------------------------------------------- quantizer (§3.2)

    /// <summary>Gets or sets the bandwidths (kbps) trained, one drawn per step (1.5, 3, 6, 12, 24 at 24 kHz).</summary>
    public double[] TargetBandwidths { get; set; } = [1.5, 3.0, 6.0, 12.0, 24.0];

    /// <summary>Gets or sets the codebooks' EMA decay (0.99).</summary>
    public double CodebookDecay { get; set; } = 0.99;

    /// <summary>Gets or sets the k-means iterations of the first-batch codebook initialization (50; reference).</summary>
    public int KMeansIterations { get; set; } = 50;

    /// <summary>Gets or sets the EMA usage below which a code is replaced by a batch vector (2; reference).</summary>
    public int DeadCodeThreshold { get; set; } = 2;

    /// <summary>Gets or sets whether the commitment loss is the mean over codebooks of the mean squared error (the
    /// reference code) rather than Eq. 3's sum over codebooks of squared norms (false).</summary>
    public bool CommitmentAsMean { get; set; }

    // ---------------------------------------------------------------- losses and balancer (§3.4)

    /// <summary>Gets or sets λ_t, the time-domain L1 weight (0.1).</summary>
    public double TimeLossWeight { get; set; } = 0.1;

    /// <summary>Gets or sets λ_f, the multi-scale mel weight (1).</summary>
    public double FrequencyLossWeight { get; set; } = 1.0;

    /// <summary>Gets or sets λ_g, the adversarial weight (3 at 24 kHz, 4 at 48 kHz).</summary>
    public double AdversarialLossWeight { get; set; } = 3.0;

    /// <summary>Gets or sets λ_feat, the relative feature-matching weight (3 at 24 kHz, 4 at 48 kHz).</summary>
    public double FeatureLossWeight { get; set; } = 3.0;

    /// <summary>Gets or sets λ_w, the commitment weight, applied outside the balancer (1).</summary>
    public double CommitmentLossWeight { get; set; } = 1.0;

    /// <summary>Gets or sets whether the balancer rescales the losses' gradients (true); false backpropagates
    /// Σ λ_i ℓ_i.</summary>
    public bool UseBalancer { get; set; } = true;

    /// <summary>Gets or sets the balancer's reference norm R (1).</summary>
    public double BalancerTotalNorm { get; set; } = 1.0;

    /// <summary>Gets or sets the balancer's EMA decay β (0.999).</summary>
    public double BalancerDecay { get; set; } = 0.999;

    /// <summary>Gets or sets the mel losses' scales i (window 2^i, hop 2^i / 4): 5 … 11.</summary>
    public int[] MelScales { get; set; } = [5, 6, 7, 8, 9, 10, 11];

    /// <summary>Gets or sets the mel bins (64).</summary>
    public int MelBins { get; set; } = 64;

    /// <summary>Gets or sets the mel losses' lowest frequency (64 Hz; the authors' training configuration, as the paper
    /// does not state it).</summary>
    public double MelMinFrequency { get; set; } = 64.0;

    /// <summary>Gets or sets α_i, the L2 coefficient of every scale (1).</summary>
    public double MelL2Coefficient { get; set; } = 1.0;

    // ---------------------------------------------------------------- discriminator (§3.4)

    /// <summary>Gets or sets the MS-STFT discriminator's STFT windows (2048, 1024, 512, 256, 128; doubled at 48 kHz).</summary>
    public int[] DiscriminatorWindows { get; set; } = [2048, 1024, 512, 256, 128];

    /// <summary>Gets or sets the discriminator's channels (32).</summary>
    public int DiscriminatorFilters { get; set; } = 32;

    /// <summary>Gets or sets the time dilations of the strided convolutions (1, 2, 4).</summary>
    public int[] DiscriminatorDilations { get; set; } = [1, 2, 4];

    /// <summary>Gets or sets the kernel along time (3).</summary>
    public int DiscriminatorKernelTime { get; set; } = 3;

    /// <summary>Gets or sets the kernel along frequency (9: the paper's Fig. 2 and the reference; the text says 8).</summary>
    public int DiscriminatorKernelFrequency { get; set; } = 9;

    /// <summary>Gets or sets the discriminator's LeakyReLU slope (0.2; reference).</summary>
    public double DiscriminatorSlope { get; set; } = 0.2;

    /// <summary>Gets or sets whether each bandwidth has its own discriminator (true: "a dedicated discriminator
    /// per-bandwidth").</summary>
    public bool DiscriminatorPerBandwidth { get; set; } = true;

    /// <summary>Gets or sets the probability a step updates the discriminator (2/3 at 24 kHz, 0.5 at 48 kHz).</summary>
    public double DiscriminatorUpdateProbability { get; set; } = 2.0 / 3.0;

    // ---------------------------------------------------------------- optimizer (§4.3)

    /// <summary>Gets or sets Adam's learning rate (3e-4).</summary>
    public double LearningRate { get; set; } = 3e-4;

    /// <summary>Gets or sets Adam's β1 (0.5).</summary>
    public double Beta1 { get; set; } = 0.5;

    /// <summary>Gets or sets Adam's β2 (0.9).</summary>
    public double Beta2 { get; set; } = 0.9;

    // ---------------------------------------------------------------- presets

    /// <summary>The configuration of the released 24 kHz checkpoint (<c>encodec_24khz</c>): the paper's defaults with the
    /// residual kernels (3, 1).</summary>
    public static EnCodecOptions OfficialCheckpoint24kHz() => new() { ResidualKernelSizes = [3, 1] };

    /// <summary>The configuration of the released 48 kHz stereo checkpoint (<c>encodec_48khz</c>): non-streamable,
    /// layer-normalized, 1 s chunks with volume normalization, 16 codebooks (3 to 24 kbps), λ_g = λ_feat = 4, doubled
    /// discriminator windows updated with probability 0.5, residual kernels (3, 1).</summary>
    public static EnCodecOptions OfficialCheckpoint48kHz() => new()
    {
        SampleRate = 48000,
        Channels = 2,
        NumQuantizers = 16,
        TargetBandwidths = [3.0, 6.0, 12.0, 24.0],
        TargetBandwidthKbps = 6.0,
        SegmentSize = 48000,
        Causal = false,
        NormalizeVolume = true,
        ChunkSeconds = 1.0,
        ResidualKernelSizes = [3, 1],
        AdversarialLossWeight = 4.0,
        FeatureLossWeight = 4.0,
        DiscriminatorWindows = [4096, 2048, 1024, 512, 256],
        DiscriminatorUpdateProbability = 0.5,
    };
}
