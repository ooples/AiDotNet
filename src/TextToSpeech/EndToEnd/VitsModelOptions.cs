namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>Options shared by the VITS family (VITS, VITS2 and the models built on them): the text encoder, the posterior
/// encoder, the flow, the HiFi-GAN decoder and discriminators, the training objective and the optimizer.</summary>
/// <remarks>
/// <para>
/// Defaults are VITS's (Kim et al. 2021, §3–4, App. B) with the values it leaves to its reference implementation
/// (jaywalnut310/vits <c>configs/ljs_base.json</c>, <c>models.py</c>): text encoder width 192, 6 relative-position layers,
/// 2 heads, window 4, filter 768, kernel 3, dropout 0.1; latent 192; posterior encoder of 16 WaveNet layers (kernel 5);
/// flow of 4 residual couplings with 4 WaveNet layers (kernel 5); HiFi-GAN V1 decoder (512 channels, rates 8, 8, 2, 2,
/// kernels 16, 16, 4, 4, resblocks 3, 7, 11 with dilations 1, 3, 5); periods 2, 3, 5, 7, 11 plus one scale
/// discriminator; 8192-sample decoder segments; λ_mel = 45, λ_kl = 1; AdamW (β = 0.8, 0.99, ε = 1e-9, weight decay 0.01)
/// at 2e-4 decayed 0.999875 per epoch; 22.05 kHz, 1024-point FFT, hop 256, 80 mel bins; prior noise scale 0.667 at
/// synthesis; a 256-wide speaker table when there are several speakers.
/// </para>
/// <para><b>For Beginners:</b> These options configure a VITS-style model. Default values follow the original paper settings.</para>
/// </remarks>
public abstract class VitsModelOptions : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    protected VitsModelOptions(VitsModelOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        Epsilon = other.Epsilon;
        EncoderKernelSize = other.EncoderKernelSize;
        RelativeWindow = other.RelativeWindow;
        PosteriorLayers = other.PosteriorLayers;
        PosteriorKernelSize = other.PosteriorKernelSize;
        FlowLayers = other.FlowLayers;
        FlowKernelSize = other.FlowKernelSize;
        DurationPredictorKernelSize = other.DurationPredictorKernelSize;
        DurationPredictorDropout = other.DurationPredictorDropout;
        ResblockType = other.ResblockType;
        UpsampleRates = (int[])other.UpsampleRates.Clone();
        UpsampleKernelSizes = (int[])other.UpsampleKernelSizes.Clone();
        UpsampleInitialChannels = other.UpsampleInitialChannels;
        ResblockKernelSizes = (int[])other.ResblockKernelSizes.Clone();
        ResblockDilationSizes = other.ResblockDilationSizes.Select(d => (int[])d.Clone()).ToArray();
        DiscriminatorPeriods = (int[])other.DiscriminatorPeriods.Clone();
        DiscriminatorWidthDivisor = other.DiscriminatorWidthDivisor;
        SegmentSize = other.SegmentSize;
        WindowSize = other.WindowSize;
        MelLossWeight = other.MelLossWeight;
        KlLossWeight = other.KlLossWeight;
        LearningRateDecay = other.LearningRateDecay;
        UpdatesPerEpoch = other.UpdatesPerEpoch;
        AddBlank = other.AddBlank;
        NumSpeakers = other.NumSpeakers;
        SpeakerEmbeddingDim = other.SpeakerEmbeddingDim;
        NumLanguages = other.NumLanguages;
        LanguageEmbeddingDim = other.LanguageEmbeddingDim;
        NoiseScale = other.NoiseScale;
        LengthScale = other.LengthScale;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Creates the shared VITS configuration.</summary>
    protected VitsModelOptions()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        FftSize = 1024;
        HiddenDim = 192;
        InterChannels = 192;
        FilterChannels = 768;
        NumHeads = 2;
        NumEncoderLayers = 6;
        DropoutRate = 0.1;
        NumFlowSteps = 4;
        LearningRate = 2e-4;
        WeightDecay = 0.01;
    }

    /// <summary>Gets or sets AdamW's β₁ (0.8).</summary>
    public double Beta1 { get; set; } = 0.8;

    /// <summary>Gets or sets AdamW's β₂ (0.99).</summary>
    public double Beta2 { get; set; } = 0.99;

    /// <summary>Gets or sets AdamW's ε (1e-9).</summary>
    public double Epsilon { get; set; } = 1e-9;

    /// <summary>Gets or sets the text encoder's feed-forward kernel (3).</summary>
    public int EncoderKernelSize { get; set; } = 3;

    /// <summary>Gets or sets the text encoder attention's maximum relative position (4).</summary>
    public int RelativeWindow { get; set; } = 4;

    /// <summary>Gets or sets the posterior encoder's WaveNet layers (16).</summary>
    public int PosteriorLayers { get; set; } = 16;

    /// <summary>Gets or sets the posterior encoder's WaveNet kernel (5).</summary>
    public int PosteriorKernelSize { get; set; } = 5;

    /// <summary>Gets or sets the WaveNet layers in each flow coupling (4).</summary>
    public int FlowLayers { get; set; } = 4;

    /// <summary>Gets or sets the flow couplings' WaveNet kernel (5).</summary>
    public int FlowKernelSize { get; set; } = 5;

    /// <summary>Gets or sets the duration predictor's kernel (3).</summary>
    public int DurationPredictorKernelSize { get; set; } = 3;

    /// <summary>Gets or sets the duration predictor's dropout (0.5).</summary>
    public double DurationPredictorDropout { get; set; } = 0.5;

    /// <summary>Gets or sets the decoder's residual block type (1: HiFi-GAN V1's three-layer blocks; 2: the
    /// one-convolution-per-dilation blocks of HiFi-GAN V2/V3).</summary>
    public int ResblockType { get; set; } = 1;

    /// <summary>Gets or sets the decoder's upsampling rates (8, 8, 2, 2; their product is the hop).</summary>
    public int[] UpsampleRates { get; set; } = [8, 8, 2, 2];

    /// <summary>Gets or sets the decoder's upsampling kernels (16, 16, 4, 4).</summary>
    public int[] UpsampleKernelSizes { get; set; } = [16, 16, 4, 4];

    /// <summary>Gets or sets the decoder's first width (512).</summary>
    public int UpsampleInitialChannels { get; set; } = 512;

    /// <summary>Gets or sets the decoder's residual block kernels (3, 7, 11).</summary>
    public int[] ResblockKernelSizes { get; set; } = [3, 7, 11];

    /// <summary>Gets or sets the decoder's residual block dilations ((1, 3, 5) for each kernel).</summary>
    public int[][] ResblockDilationSizes { get; set; } = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];

    /// <summary>Gets or sets the period discriminators' periods (2, 3, 5, 7, 11).</summary>
    public int[] DiscriminatorPeriods { get; set; } = [2, 3, 5, 7, 11];

    /// <summary>Gets or sets a divisor on every discriminator width (1: the paper's widths).</summary>
    public int DiscriminatorWidthDivisor { get; set; } = 1;

    /// <summary>Gets or sets the decoder's training segment in samples (8192).</summary>
    public int SegmentSize { get; set; } = 8192;

    /// <summary>Gets or sets the STFT window (1024).</summary>
    public int WindowSize { get; set; } = 1024;

    /// <summary>Gets or sets λ_mel (45).</summary>
    public double MelLossWeight { get; set; } = 45.0;

    /// <summary>Gets or sets λ_kl (1).</summary>
    public double KlLossWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the per-epoch learning-rate decay (0.999875).</summary>
    public double LearningRateDecay { get; set; } = 0.999875;

    /// <summary>Gets or sets the updates in one epoch (196: LJSpeech's 12,500 training clips at a batch of 64).</summary>
    public int UpdatesPerEpoch { get; set; } = 196;

    /// <summary>Gets or sets whether a blank token (0) is placed between and around the characters.</summary>
    public bool AddBlank { get; set; } = true;

    /// <summary>Gets or sets the number of speakers; above one, a speaker table conditions the model and every training
    /// sample and synthesis names its speaker (0: a single-speaker model).</summary>
    public int NumSpeakers { get; set; }

    /// <summary>Gets or sets the width of the speaker embedding (256, the reference's <c>gin_channels</c> for VCTK).</summary>
    public int SpeakerEmbeddingDim { get; set; } = 256;

    /// <summary>Gets or sets the number of languages; above one, a language embedding is concatenated to every character
    /// embedding and conditions the duration predictor, and every training sample and synthesis names its language
    /// (YourTTS §2; 0: a monolingual model).</summary>
    public int NumLanguages { get; set; }

    /// <summary>Gets or sets the width of the language embedding (4, YourTTS §2).</summary>
    public int LanguageEmbeddingDim { get; set; } = 4;

    /// <summary>Gets or sets the prior noise scale at synthesis (0.667).</summary>
    public double NoiseScale { get; set; } = 0.667;

    /// <summary>Gets or sets the duration multiplier at synthesis (1).</summary>
    public double LengthScale { get; set; } = 1.0;

    /// <summary>Gets or sets the seed of the training draws and the synthesis noise, for repeatable runs.</summary>
    public int SamplingSeed { get; set; }
}
