namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Native Conformer/U-Net flow and S3 encoder settings used by CosyVoice v1/v2 and Chatterbox.</summary>
public abstract class CosyVoiceConformerOptions : CosyVoiceFamilyOptions
{
    /// <summary>Creates the released common encoder defaults.</summary>
    protected CosyVoiceConformerOptions() { }

    /// <summary>Copies the native Conformer flow and tokenizer settings.</summary>
    protected CosyVoiceConformerOptions(CosyVoiceConformerOptions other) : base(other)
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
        SpeechTokenizerState = other.SpeechTokenizerState;
        SpeechTokenizerHeads = other.SpeechTokenizerHeads;
        SpeechTokenizerLayers = other.SpeechTokenizerLayers;
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

    /// <summary>The S3 speech tokenizer's width (1,280 in every release).</summary>
    public int SpeechTokenizerState { get; set; } = 1280;

    /// <summary>The S3 speech tokenizer's attention heads.</summary>
    public int SpeechTokenizerHeads { get; set; } = 20;

    /// <summary>The S3 speech tokenizer's blocks.</summary>
    public int SpeechTokenizerLayers { get; set; } = 6;
}
