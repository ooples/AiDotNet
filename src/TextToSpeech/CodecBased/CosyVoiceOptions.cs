namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for CosyVoice.</summary>
/// <remarks>
/// <para>The defaults are CosyVoice-300M's (FunAudioLLM/CosyVoice <c>examples/libritts/cosyvoice/conf/cosyvoice.yaml</c>):
/// Whisper's 100-language text tokens (51,866) and the 50 Hz S3 speech tokens (4,096); the LM's 6-block Conformer text
/// encoder (512 in, 1,024 wide, 16 heads, 4,096 units) and 14-block Transformer (1,024 wide); the flow's 6-block
/// Conformer (512 wide, 8 heads, 2,048 units) and U-Net estimator (channels 256, 256; 4 blocks; 12 mid blocks; 8 heads
/// of 64), 10 Euler steps with guidance 0.7; HiFT at 22.05 kHz (base 512, 8 harmonics, upsampling 8·8, iSTFT 16/4); a
/// 192-wide CAM++ x-vector.</para>
/// <para><b>For Beginners:</b> These options configure the CosyVoice model. Default values follow the released model.</para>
/// </remarks>
public class CosyVoiceOptions : CodecTtsOptions
{
    public CosyVoiceOptions()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        FftSize = 1024;
        NumCodebooks = 1;
        CodebookSize = 4096;
        CodecFrameRate = 50;
        VocabSize = 51866;
        TextEncoderDim = 512;
        LLMDim = 1024;
        NumEncoderLayers = 6;
        NumLLMLayers = 14;
        NumHeads = 16;
        SpeakerEmbeddingDim = 192;
        DropoutRate = 0.1;
    }

    /// <summary>Initializes a new instance by copying all CosyVoice settings.</summary>
    public CosyVoiceOptions(CosyVoiceOptions other)
        : base(other)
    {
        TextLanguages = other.TextLanguages;
        LmFeedForwardDim = other.LmFeedForwardDim;
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
    }

    /// <summary>Language tokens in the text tokenizer: 100 (Whisper large-v3's) for CosyVoice-300M.</summary>
    public int TextLanguages { get; set; } = 100;

    /// <summary>The LM's feed-forward width (<c>linear_units</c>), in the text encoder and the Transformer alike.</summary>
    public int LmFeedForwardDim { get; set; } = 4096;

    /// <summary>The flow's width (<c>input_size</c>, its encoder's <c>output_size</c>).</summary>
    public int FlowDim { get; set; } = 512;

    /// <summary>The flow encoder's attention heads.</summary>
    public int FlowHeads { get; set; } = 8;

    /// <summary>The flow encoder's feed-forward width.</summary>
    public int FlowFeedForwardDim { get; set; } = 2048;

    /// <summary>The flow encoder's Conformer blocks.</summary>
    public int FlowEncoderLayers { get; set; } = 6;

    /// <summary>The flow encoder's attention dropout (0.1, where the LM's is 0).</summary>
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

    /// <summary>The LM sampler's nucleus (<c>top_p</c>).</summary>
    public double TopP { get; set; } = 0.8;

    /// <summary>The LM sampler's candidate count (<c>top_k</c>).</summary>
    public int TopK { get; set; } = 25;

    /// <summary>The repetition-aware sampler's window (<c>win_size</c>).</summary>
    public int RepetitionWindow { get; set; } = 10;

    /// <summary>The repetition-aware sampler's threshold ratio (<c>tau_r</c>).</summary>
    public double RepetitionRatio { get; set; } = 0.1;

    /// <summary>The S3 speech tokenizer's width (<c>speech_tokenizer_v1</c>: 1,280).</summary>
    public int SpeechTokenizerState { get; set; } = 1280;

    /// <summary>The S3 speech tokenizer's attention heads.</summary>
    public int SpeechTokenizerHeads { get; set; } = 20;

    /// <summary>The S3 speech tokenizer's blocks.</summary>
    public int SpeechTokenizerLayers { get; set; } = 6;

    /// <summary>CAM++'s TDNN channels (<c>init_channels</c>).</summary>
    public int SpeakerEncoderChannels { get; set; } = 128;

    /// <summary>CAM++'s D-TDNN growth rate.</summary>
    public int SpeakerEncoderGrowthRate { get; set; } = 32;

    /// <summary>The learning-rate warmup of each stage's Adam (WarmupLR's <c>warmup_steps</c>: 2,500).</summary>
    public int WarmupSteps { get; set; } = 2500;

    /// <summary>The seed of sampling and of training's random draws.</summary>
    public int SamplingSeed { get; set; } = 1986;
}
