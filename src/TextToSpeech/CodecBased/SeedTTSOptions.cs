namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The independently trainable part of the Seed-TTS pipeline.</summary>
public enum SeedTTSTrainingComponent
{
    /// <summary>The causal transformer that predicts speech tokens from text and an optional speech prompt.</summary>
    TokenLanguageModel,

    /// <summary>The coarse-to-fine diffusion transformer that turns speech tokens into continuous vocoder features.</summary>
    TokenDiffusion,

    /// <summary>The separately trained acoustic vocoder.</summary>
    AcousticVocoder,

    /// <summary>The paper's timbre-factorization self-distillation objective.</summary>
    TimbreSelfDistillation,

    /// <summary>End-to-end adversarial speech-tokenizer pretraining before token-LM training.</summary>
    SpeechTokenizer,
}

/// <summary>Options for the disclosed Seed-TTS autoregressive pipeline.</summary>
/// <remarks>
/// <para>
/// The Seed-TTS technical report discloses the four model blocks and their objectives, but intentionally does not
/// publish widths, depths, tokenizer configuration, source code, or weights. Consequently, the values below are
/// resource-conscious defaults for AiDotNet's native implementation, not purported ByteDance hyperparameters.
/// Dimensions and the native diffusion schedule are configurable; undisclosed private block details cannot be
/// certified by these options or by loading an AiDotNet-format checkpoint.
/// </para>
/// </remarks>
public class SeedTTSOptions : CodecTtsOptions
{
    /// <summary>Creates the native reference configuration.</summary>
    public SeedTTSOptions()
    {
        SampleRate = 24000;
        MelChannels = 80;
        NumCodebooks = 1;
        CodebookSize = 1024;
        CodecFrameRate = 50;
        LLMDim = 256;
        TextEncoderDim = 256;
        NumLLMLayers = 4;
        NumHeads = 4;
        NumEncoderLayers = 4;
        SpeakerEmbeddingDim = 128;
        VocabSize = 256;
        MaxCodecFrames = 512;
    }

    /// <summary>Creates a copy.</summary>
    public SeedTTSOptions(SeedTTSOptions other)
        : base(other)
    {
        Validation.Guard.NotNull(other);
        AcousticDimension = other.AcousticDimension;
        DiffusionLayers = other.DiffusionLayers;
        DiffusionSteps = other.DiffusionSteps;
        DiffusionTrainingSteps = other.DiffusionTrainingSteps;
        DiffusionBetaStart = other.DiffusionBetaStart;
        DiffusionBetaEnd = other.DiffusionBetaEnd;
        VocoderHiddenDimension = other.VocoderHiddenDimension;
        MaxPromptCodecFrames = other.MaxPromptCodecFrames;
        FramesPerTextToken = other.FramesPerTextToken;
        MinimumGeneratedCodecFrames = other.MinimumGeneratedCodecFrames;
        Temperature = other.Temperature;
        TopK = other.TopK;
        GenerationSeed = other.GenerationSeed;
        TrainingComponent = other.TrainingComponent;
        TokenizerChannels = other.TokenizerChannels;
        TokenizerCodeDimension = other.TokenizerCodeDimension;
        CodecRates = (int[])other.CodecRates.Clone();
        TrainingMelWindows = (int[])other.TrainingMelWindows.Clone();
        TrainingMelBins = (int[])other.TrainingMelBins.Clone();
        DiscriminatorWindows = (int[])other.DiscriminatorWindows.Clone();
        DiscriminatorChannels = other.DiscriminatorChannels;
        DiscriminatorWidthDivisor = other.DiscriminatorWidthDivisor;
    }

    /// <summary>Continuous width produced by the tokenizer and consumed by diffusion and the vocoder.</summary>
    public int AcousticDimension { get; set; } = 64;

    /// <summary>Configurable DAC-family tokenizer encoder width; Seed-TTS does not disclose its tokenizer widths.</summary>
    public int TokenizerChannels { get; set; } = 32;
    /// <summary>Gets or sets the feature width of the tokenizer code. Defaults to <c>8</c>.</summary>
    public int TokenizerCodeDimension { get; set; } = 8;
    /// <summary>Gets or sets the codec rates.</summary>
    public int[] CodecRates { get; set; } = [2, 4, 5, 12];
    /// <summary>Gets or sets the training mel windows.</summary>
    public int[] TrainingMelWindows { get; set; } = [32, 64, 128, 256, 512, 1024, 2048];
    /// <summary>Gets or sets the training mel bins.</summary>
    public int[] TrainingMelBins { get; set; } = [5, 10, 20, 40, 80, 160, 320];
    /// <summary>Gets or sets the discriminator windows.</summary>
    public int[] DiscriminatorWindows { get; set; } = [2048, 1024, 512];
    /// <summary>Gets or sets the discriminator channels. Defaults to <c>32</c>.</summary>
    public int DiscriminatorChannels { get; set; } = 32;
    /// <summary>Gets or sets the discriminator width divisor. Defaults to <c>1</c>.</summary>
    public int DiscriminatorWidthDivisor { get; set; } = 1;

    /// <summary>Number of bidirectional transformer blocks in the token diffusion model.</summary>
    public int DiffusionLayers { get; set; } = 4;

    /// <summary>Number of coarse-to-fine denoising iterations at inference.</summary>
    public int DiffusionSteps { get; set; } = 8;

    /// <summary>Native VP diffusion training schedule length; the Seed report does not disclose a schedule.</summary>
    public int DiffusionTrainingSteps { get; set; } = 1000;

    /// <summary>First native linear VP beta, following the standard DDPM configuration, not a reported Seed value.</summary>
    public double DiffusionBetaStart { get; set; } = 0.0001;

    /// <summary>Last native linear VP beta; inference uses deterministic DDIM with clean-feature prediction.</summary>
    public double DiffusionBetaEnd { get; set; } = 0.02;

    /// <summary>Hidden width of the separately trained waveform decoder.</summary>
    public int VocoderHiddenDimension { get; set; } = 128;

    /// <summary>Maximum number of prompt speech tokens retained for in-context continuation.</summary>
    public int MaxPromptCodecFrames { get; set; } = 400;

    /// <summary>Generation budget per target text token when an end token has not yet been emitted.</summary>
    public int FramesPerTextToken { get; set; } = 3;

    /// <summary>Minimum speech-token count before the autoregressive decoder may accept its end token.</summary>
    public int MinimumGeneratedCodecFrames { get; set; } = 4;

    /// <summary>Autoregressive sampling temperature.</summary>
    public double Temperature { get; set; } = 1.0;

    /// <summary>Top-k speech-token sampling; zero uses the entire vocabulary.</summary>
    public int TopK { get; set; } = 50;

    /// <summary>Seed used by the speech-token sampler and diffusion initial noise.</summary>
    public int GenerationSeed { get; set; } = 2406;

    /// <summary>The component optimized by <c>Train(TtsTrainingSample)</c>.</summary>
    public SeedTTSTrainingComponent TrainingComponent { get; set; } = SeedTTSTrainingComponent.TokenLanguageModel;

    /// <summary>Samples represented by one speech token in this native configuration.</summary>
    public int SamplesPerCodecFrame => SampleRate / CodecFrameRate;
}
