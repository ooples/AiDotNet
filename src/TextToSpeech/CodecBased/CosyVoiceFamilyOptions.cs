using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// Options shared by the CosyVoice models: audio and codec geometry, language-model dimensions, HiFT,
/// CAM++ conditioning, optional released frontend paths and training. Each model sets its release's defaults.
/// </summary>
[AiDotNet.Attributes.DimensionDivisibility(nameof(LLMDim), nameof(NumHeads))]
public abstract class CosyVoiceFamilyOptions : ModelOptions
{
    /// <summary>Samples per mel frame; must equal four times the product of HiFT upsampling rates.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("HiFT fixes its iSTFT hop at four, so HopSize must vary with HiftUpsampleRates. CosyVoiceOptionConstraintTests rejects unpaired changes and verifies the actual waveform length for valid pairs.")]
    public int HopSize { get; set; } = 256;

    /// <summary>The single S3 token stream used by the CosyVoice family.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("CosyVoice emits one stream of S3 tokens rather than RVQ codebooks. Its constructor requires NumCodebooks = 1; CosyVoiceOptionConstraintTests verifies that other counts are rejected.")]
    public int NumCodebooks { get; set; } = 8;

    /// <summary>Audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 22050;
    /// <summary>Mel-spectrogram frequency channels.</summary>
    public int MelChannels { get; set; } = 80;
    /// <summary>FFT window size for the conditioning frontend.</summary>
    public int FftSize { get; set; } = 1024;
    /// <summary>Language-model query attention heads.</summary>
    public int NumHeads { get; set; } = 8;
    /// <summary>Text vocabulary size.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("CosyVoice v1 and its voice-cloning wrapper derive this vocabulary from the fixed Whisper tokenizer; v2 and v3 have separate configurable language models.", ModelType = typeof(CosyVoice<>))]
    public int VocabSize { get; set; } = 256;
    /// <summary>Maximum input text length.</summary>
    public int MaxTextLength { get; set; } = 512;
    /// <summary>Native encoder dropout probability.</summary>
    public double DropoutRate { get; set; } = 0.1;
    /// <summary>Exported graph runtime settings.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    /// <summary>Configured language-model and flow learning rate; the GAN retains its separately declared rate.</summary>
    public double LearningRate { get; set; } = 1e-4;
    /// <summary>Optional coupled L2 weight decay for native Adam (zero in the released recipes).</summary>
    public double WeightDecay { get; set; } = 0.0;
    /// <summary>Speech-token vocabulary cardinality.</summary>
    [AiDotNet.Attributes.NotScaled("the codec's codes per codebook, which it must equal")]
    public int CodebookSize { get; set; } = 1024;
    /// <summary>Speech tokens per second.</summary>
    public int CodecFrameRate { get; set; } = 50;
    /// <summary>Language-model hidden width.</summary>
    public int LLMDim { get; set; } = 1024;
    /// <summary>Language-model decoder blocks.</summary>
    public int NumLLMLayers { get; set; } = 12;
    /// <summary>Flow conditioning x-vector width.</summary>
    public int SpeakerEmbeddingDim { get; set; } = 256;

    /// <summary>Creates an instance with the default settings.</summary>
    protected CosyVoiceFamilyOptions() { }

    /// <summary>Copies every shared setting.</summary>
    protected CosyVoiceFamilyOptions(CosyVoiceFamilyOptions other)
        : base(other)
    {
        SampleRate = other.SampleRate;
        MelChannels = other.MelChannels;
        HopSize = other.HopSize;
        FftSize = other.FftSize;
        NumHeads = other.NumHeads;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        DropoutRate = other.DropoutRate;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        NumCodebooks = other.NumCodebooks;
        CodebookSize = other.CodebookSize;
        CodecFrameRate = other.CodecFrameRate;
        LLMDim = other.LLMDim;
        NumLLMLayers = other.NumLLMLayers;
        SpeakerEmbeddingDim = other.SpeakerEmbeddingDim;
        HiftBaseChannels = other.HiftBaseChannels;
        HiftHarmonics = other.HiftHarmonics;
        HiftUpsampleRates = (int[])other.HiftUpsampleRates.Clone();
        HiftUpsampleKernels = (int[])other.HiftUpsampleKernels.Clone();
        HiftSourceKernels = (int[])other.HiftSourceKernels.Clone();
        SpeakerEncoderChannels = other.SpeakerEncoderChannels;
        SpeakerEncoderGrowthRate = other.SpeakerEncoderGrowthRate;
        SamplingSeed = other.SamplingSeed;
        WarmupSteps = other.WarmupSteps;
        IncludeDiscriminators = other.IncludeDiscriminators;
        VocoderSegmentLength = other.VocoderSegmentLength;
        MaxReferenceSeconds = other.MaxReferenceSeconds;
        SpeechTokenizerPath = other.SpeechTokenizerPath;
        SpeakerEncoderOnnxPath = other.SpeakerEncoderOnnxPath;
    }

    /// <summary>Maximum prompt duration accepted by the released frontend (30 seconds).</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("This limit applies when the optional released ONNX speech frontend is configured. ChatterboxOptionContextTests loads a contract graph, verifies rejection at a one-second limit and executes the same two-second input at a two-second limit; native fixtures without the graph do not exercise this boundary.")]
    public int MaxReferenceSeconds { get; set; } = 30;

    /// <summary>Optional original frozen speech-tokenizer ONNX graph; release loading sets this path.</summary>
    public string? SpeechTokenizerPath { get; set; }

    /// <summary>Optional original frozen CAM++ graph; release loading sets this path.</summary>
    public string? SpeakerEncoderOnnxPath { get; set; }

    /// <summary>HiFT's first upsampling width (<c>base_channels</c>).</summary>
    public int HiftBaseChannels { get; set; } = 512;

    /// <summary>HiFT's harmonics above the fundamental (<c>nb_harmonics</c>).</summary>
    public int HiftHarmonics { get; set; } = 8;

    /// <summary>HiFT's upsampling rates; their product times the iSTFT hop of 4 is <c>HopSize</c>.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("HiFT upsampling rates must be paired with HopSize and per-stage kernels. CosyVoiceOptionConstraintTests and ChatterboxOptionContextTests verify native waveform lengths for valid pairs; unpaired hop geometry is rejected.")]
    public int[] HiftUpsampleRates { get; set; } = { 8, 8 };

    /// <summary>HiFT's transposed-convolution kernels, one per upsampling stage.</summary>
    public int[] HiftUpsampleKernels { get; set; } = { 16, 16 };

    /// <summary>HiFT's source-branch residual block kernels, one per upsampling stage.</summary>
    public int[] HiftSourceKernels { get; set; } = { 7, 11 };

    /// <summary>CAM++'s TDNN channels (<c>init_channels</c>).</summary>
    public int SpeakerEncoderChannels { get; set; } = 128;

    /// <summary>CAM++'s D-TDNN growth rate.</summary>
    public int SpeakerEncoderGrowthRate { get; set; } = 32;

    /// <summary>The learning-rate warmup of each stage's Adam (the recipe's <c>warmup_steps</c>; 0 for a constant rate).</summary>
    public int WarmupSteps { get; set; } = 2500;

    /// <summary>Whether to build HiFT's discriminators, which only <see cref="CosyVoiceTrainingStage.Vocoder"/> training uses.</summary>
    public bool IncludeDiscriminators { get; set; }

    /// <summary>The samples a vocoder training step crops or pads an utterance to (the recipe's <c>truncate_length</c>); a
    /// multiple of HopSize.</summary>
    public int VocoderSegmentLength { get; set; } = 24576;

    /// <summary>The seed of sampling and of training's random draws.</summary>
    public int SamplingSeed { get; set; } = 1986;
}
