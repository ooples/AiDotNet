namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// Options shared by the CosyVoice models: the flow (encoder, U-Net estimator, sampling), the HiFT vocoder, the frozen
/// S3 tokenizer and CAM++ front end, the language model's sampling, and training. Each model's options set its release's
/// values.
/// </summary>
public abstract class CosyVoiceFamilyOptions : CodecTtsOptions
{
    protected CosyVoiceFamilyOptions() { }

    /// <summary>Copies every shared setting.</summary>
    protected CosyVoiceFamilyOptions(CosyVoiceFamilyOptions other)
        : base(other)
    {
        FlowDim = other.FlowDim;
        FlowHeads = other.FlowHeads;
        FlowFeedForwardDim = other.FlowFeedForwardDim;
        FlowEncoderLayers = other.FlowEncoderLayers;
        FlowAttentionDropoutRate = other.FlowAttentionDropoutRate;
        DecoderChannels = (int[])other.DecoderChannels.Clone();
        DecoderHeads = other.DecoderHeads;
        DecoderHeadDim = other.DecoderHeadDim;
        DecoderBlocks = other.DecoderBlocks;
        DecoderMidBlocks = other.DecoderMidBlocks;
        FlowTimesteps = other.FlowTimesteps;
        TrainingCfgRate = other.TrainingCfgRate;
        InferenceCfgRate = other.InferenceCfgRate;
        FlowLoss = other.FlowLoss;
        HiftBaseChannels = other.HiftBaseChannels;
        HiftHarmonics = other.HiftHarmonics;
        HiftUpsampleRates = (int[])other.HiftUpsampleRates.Clone();
        HiftUpsampleKernels = (int[])other.HiftUpsampleKernels.Clone();
        HiftSourceKernels = (int[])other.HiftSourceKernels.Clone();
        TopP = other.TopP;
        TopK = other.TopK;
        RepetitionWindow = other.RepetitionWindow;
        RepetitionRatio = other.RepetitionRatio;
        SpeechTokenizerState = other.SpeechTokenizerState;
        SpeechTokenizerHeads = other.SpeechTokenizerHeads;
        SpeechTokenizerLayers = other.SpeechTokenizerLayers;
        SpeakerEncoderChannels = other.SpeakerEncoderChannels;
        SpeakerEncoderGrowthRate = other.SpeakerEncoderGrowthRate;
        SamplingSeed = other.SamplingSeed;
        WarmupSteps = other.WarmupSteps;
        IncludeDiscriminators = other.IncludeDiscriminators;
        VocoderSegmentLength = other.VocoderSegmentLength;
    }

    /// <summary>The flow's width (<c>input_size</c>, its encoder's <c>output_size</c>).</summary>
    public int FlowDim { get; set; } = 512;

    /// <summary>The flow encoder's attention heads.</summary>
    public int FlowHeads { get; set; } = 8;

    /// <summary>The flow encoder's feed-forward width.</summary>
    public int FlowFeedForwardDim { get; set; } = 2048;

    /// <summary>The flow encoder's Conformer blocks.</summary>
    public int FlowEncoderLayers { get; set; } = 6;

    /// <summary>The flow encoder's attention dropout.</summary>
    public double FlowAttentionDropoutRate { get; set; } = 0.1;

    /// <summary>The flow estimator's U-Net channels.</summary>
    public int[] DecoderChannels { get; set; } = { 256, 256 };

    /// <summary>The estimator's attention heads.</summary>
    public int DecoderHeads { get; set; } = 8;

    /// <summary>The estimator's attention head width.</summary>
    public int DecoderHeadDim { get; set; } = 64;

    /// <summary>The estimator's transformer blocks per U-Net level.</summary>
    public int DecoderBlocks { get; set; } = 4;

    /// <summary>The estimator's middle blocks.</summary>
    public int DecoderMidBlocks { get; set; } = 12;

    /// <summary>Euler steps of the flow's sampler.</summary>
    public int FlowTimesteps { get; set; } = 10;

    /// <summary>The probability a flow training step drops its conditions (classifier-free guidance training).</summary>
    public double TrainingCfgRate { get; set; } = 0.2;

    /// <summary>The flow's guidance weight at inference.</summary>
    public double InferenceCfgRate { get; set; } = 0.7;

    /// <summary>The flow-matching loss: the paper's absolute error (eq. 10) or the code's squared error.</summary>
    public CosyVoiceFlowLoss FlowLoss { get; set; } = CosyVoiceFlowLoss.AbsoluteError;

    /// <summary>HiFT's first upsampling width (<c>base_channels</c>).</summary>
    public int HiftBaseChannels { get; set; } = 512;

    /// <summary>HiFT's harmonics above the fundamental (<c>nb_harmonics</c>).</summary>
    public int HiftHarmonics { get; set; } = 8;

    /// <summary>HiFT's upsampling rates; their product times the iSTFT hop of 4 is <c>HopSize</c>.</summary>
    public int[] HiftUpsampleRates { get; set; } = { 8, 8 };

    /// <summary>HiFT's transposed-convolution kernels, one per upsampling stage.</summary>
    public int[] HiftUpsampleKernels { get; set; } = { 16, 16 };

    /// <summary>HiFT's source-branch residual block kernels, one per upsampling stage.</summary>
    public int[] HiftSourceKernels { get; set; } = { 7, 11 };

    /// <summary>The LM sampler's nucleus (<c>top_p</c>).</summary>
    public double TopP { get; set; } = 0.8;

    /// <summary>The LM sampler's candidate count (<c>top_k</c>).</summary>
    public int TopK { get; set; } = 25;

    /// <summary>The repetition-aware sampler's window (<c>win_size</c>).</summary>
    public int RepetitionWindow { get; set; } = 10;

    /// <summary>The repetition-aware sampler's threshold ratio (<c>tau_r</c>).</summary>
    public double RepetitionRatio { get; set; } = 0.1;

    /// <summary>The S3 speech tokenizer's width (1,280 in every release).</summary>
    public int SpeechTokenizerState { get; set; } = 1280;

    /// <summary>The S3 speech tokenizer's attention heads.</summary>
    public int SpeechTokenizerHeads { get; set; } = 20;

    /// <summary>The S3 speech tokenizer's blocks.</summary>
    public int SpeechTokenizerLayers { get; set; } = 6;

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
