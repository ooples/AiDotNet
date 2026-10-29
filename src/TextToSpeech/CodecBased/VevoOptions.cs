namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Which of Vevo's networks <see cref="Vevo{T}"/> trains.</summary>
public enum VevoStage
{
    /// <summary>Vevo-TTS's autoregressive transformer: phones to content-style tokens (the release's
    /// <c>PhoneToVq8192</c>).</summary>
    TextToContentStyle,

    /// <summary>Vevo-Voice's autoregressive transformer: content tokens or phones (half the samples each) to content-style
    /// tokens (the release's <c>Vq32ToVq8192</c>).</summary>
    ContentToContentStyle,

    /// <summary>The flow-matching transformer: content-style tokens to mel (the release's <c>Vq8192ToMels</c>).</summary>
    Acoustic,
}

/// <summary>Options for <see cref="Vevo{T}"/>.</summary>
/// <remarks>
/// <para>The defaults are the released Vevo (Zhang et al., ICLR 2025; Amphion <c>models/vc/vevo</c> and the
/// <c>amphion/Vevo</c> checkpoints): HuBERT-Large read at its 18th layer; a content-style tokenizer of 8,192 codes (RepCodec
/// over the normalized features) and a content tokenizer of 32 (the original RepCodec over the raw ones); a 1920-wide,
/// 12-layer Llama from phones to content-style tokens (Vevo-TTS) and a 1536-wide one from content tokens (Vevo-Voice); a
/// 1024-wide, 16-layer flow-matching transformer to a 128-band mel at 24 kHz, hop 480; and a 30-block Vocos.</para>
/// <para>Decoding samples at temperature 0.8, top-k 50, top-p 0.9, with at least 50 new tokens; the flow takes 32 Euler steps
/// with guidance 1 rescaled by 0.75. Training follows the release's trainers: AdamW at 2e-4 for the autoregressive models
/// (gradient norm clipped at 1) and 7.5e-5 for the flow (clipped at 0.2), an inverse-square-root schedule after 32,000
/// warm-up steps.</para>
/// <para><b>For Beginners:</b> These settings size Vevo's networks and control how it imitates a voice; the defaults
/// reproduce the released model.</para>
/// </remarks>
public class VevoOptions : TtsModelOptions
{
    /// <summary>Creates the released Vevo configuration.</summary>
    public VevoOptions()
    {
        SampleRate = 24000;
        MelChannels = 128;
        HopSize = 480;
        FftSize = 1920;
        HiddenDim = 1024;
        NumHeads = 16;
        VocabSize = 1024;
        LearningRate = 2e-4;
        WeightDecay = 0.01;
        DropoutRate = 0.0;
    }

    // ---------------------------------------------------------------- text

    /// <summary>The language of the text to speak (<c>g2p_</c>'s: zh, en, ja, ko, fr, de).</summary>
    public string Language { get; set; } = "en";

    /// <summary>MaskGCT's polyphone model (<c>poly_bert_model.onnx</c>), which Chinese text needs.</summary>
    public string? PolyphoneModelPath { get; set; }

    /// <summary>The Open JTalk dictionary (<c>open_jtalk_dic_utf_8-1.11</c>), which Japanese text needs.</summary>
    public string? OpenJTalkDictionaryPath { get; set; }

    // ---------------------------------------------------------------- HuBERT and the tokenizers

    /// <summary>The 16 kHz rate HuBERT reads.</summary>
    public int ContentSampleRate { get; set; } = 16000;

    /// <summary>HuBERT's feature-encoder widths (HuBERT-Large: seven of 512).</summary>
    public int[] HubertConvDims { get; set; } = { 512, 512, 512, 512, 512, 512, 512 };

    /// <summary>HuBERT's feature-encoder kernels.</summary>
    public int[] HubertConvKernels { get; set; } = { 10, 3, 3, 3, 3, 2, 2 };

    /// <summary>HuBERT's feature-encoder strides (×320).</summary>
    public int[] HubertConvStrides { get; set; } = { 5, 2, 2, 2, 2, 2, 2 };

    /// <summary>HuBERT's width (1024).</summary>
    public int HubertHidden { get; set; } = 1024;

    /// <summary>HuBERT's heads (16).</summary>
    public int HubertHeads { get; set; } = 16;

    /// <summary>HuBERT's feed-forward width (4096).</summary>
    public int HubertIntermediate { get; set; } = 4096;

    /// <summary>HuBERT's position-convolution kernel (128).</summary>
    public int HubertPositionKernel { get; set; } = 128;

    /// <summary>HuBERT's position-convolution groups (16).</summary>
    public int HubertPositionGroups { get; set; } = 16;

    /// <summary>The HuBERT layer whose output the tokenizers read (18); the layers above it are never run.</summary>
    public int HubertLayer { get; set; } = 18;

    /// <summary>Content-style tokens (8192).</summary>
    public int ContentStyleCodes { get; set; } = 8192;

    /// <summary>The content-style codebook's dimension (8).</summary>
    public int ContentStyleCodeDim { get; set; } = 8;

    /// <summary>The content-style tokenizer's Vocos width (384).</summary>
    public int ContentStyleVocosDim { get; set; } = 384;

    /// <summary>The content-style tokenizer's Vocos feed-forward width (2048).</summary>
    public int ContentStyleVocosIntermediate { get; set; } = 2048;

    /// <summary>The content-style tokenizer's Vocos blocks (12).</summary>
    public int ContentStyleVocosLayers { get; set; } = 12;

    /// <summary>Content tokens (32).</summary>
    public int ContentCodes { get; set; } = 32;

    /// <summary>The content tokenizer's width and code dimension (1024).</summary>
    public int ContentCodecChannels { get; set; } = 1024;

    // ---------------------------------------------------------------- the autoregressive transformers

    /// <summary>Phone vocabulary of the autoregressive transformers (1024).</summary>
    public int PhoneVocabulary { get; set; } = 1024;

    /// <summary>Vevo-TTS's transformer width (1920).</summary>
    public int TextArHidden { get; set; } = 1920;

    /// <summary>Vevo-TTS's transformer feed-forward width (7680).</summary>
    public int TextArIntermediate { get; set; } = 7680;

    /// <summary>Vevo-TTS's transformer layers (12).</summary>
    public int TextArLayers { get; set; } = 12;

    /// <summary>Vevo-TTS's transformer heads (16).</summary>
    public int TextArHeads { get; set; } = 16;

    /// <summary>Whether Vevo-Voice and Vevo-Style (the content tokenizer and the second transformer) are built.</summary>
    public bool VoiceConversion { get; set; } = true;

    /// <summary>Vevo-Voice's transformer width (1536).</summary>
    public int VoiceArHidden { get; set; } = 1536;

    /// <summary>Vevo-Voice's transformer feed-forward width (6144).</summary>
    public int VoiceArIntermediate { get; set; } = 6144;

    /// <summary>Vevo-Voice's transformer layers (12).</summary>
    public int VoiceArLayers { get; set; } = 12;

    /// <summary>Vevo-Voice's transformer heads (16).</summary>
    public int VoiceArHeads { get; set; } = 16;

    /// <summary>Sample the next token (true, the release); otherwise take the most likely.</summary>
    public bool Sample { get; set; } = true;

    /// <summary>Temperature (0.8).</summary>
    public double Temperature { get; set; } = 0.8;

    /// <summary>Top-k (50).</summary>
    public int TopK { get; set; } = 50;

    /// <summary>Top-p (0.9).</summary>
    public double TopP { get; set; } = 0.9;

    /// <summary>Repetition penalty (1, none).</summary>
    public double RepetitionPenalty { get; set; } = 1.0;

    /// <summary>New tokens before the end token is allowed (50).</summary>
    public int MinNewTokens { get; set; } = 50;

    /// <summary>The most positions a sequence reaches, input included (2000).</summary>
    public int MaxLength { get; set; } = 2000;

    /// <summary>Restrict decoding to content-style tokens and the end token. The release does not; an untrained model can
    /// otherwise emit other ids, which the flow cannot embed.</summary>
    public bool ConstrainToContentStyleTokens { get; set; }

    /// <summary>The seed of decoding's sampling and the flow's noise.</summary>
    public int SamplingSeed { get; set; } = 1234;

    // ---------------------------------------------------------------- the flow, the mel and Vocos

    /// <summary>The flow-matching transformer's layers (16).</summary>
    public int FlowLayers { get; set; } = 16;

    /// <summary>The flow-matching transformer's heads (16).</summary>
    public int FlowHeads { get; set; } = 16;

    /// <summary>How often training drops the prompt and the condition (<c>cfg_scale</c>, 0.2).</summary>
    public double FlowPromptDropProbability { get; set; } = 0.2;

    /// <summary>σ of the probability path (1e-5).</summary>
    public double FlowSigma { get; set; } = 1e-5;

    /// <summary>Euler steps (32).</summary>
    public int FlowSteps { get; set; } = 32;

    /// <summary>Guidance scale (1).</summary>
    public double Guidance { get; set; } = 1.0;

    /// <summary>The share of the rescaled guided velocity (0.75).</summary>
    public double GuidanceRescale { get; set; } = 0.75;

    /// <summary>The mel's upper edge (12 kHz).</summary>
    public double MelMaxFrequency { get; set; } = 12000;

    /// <summary>The mel's normalization mean (−4.92).</summary>
    public double MelMean { get; set; } = -4.92;

    /// <summary>The mel's normalization variance (8.14).</summary>
    public double MelVariance { get; set; } = 8.14;

    /// <summary>Vocos's width (1024).</summary>
    public int VocoderDim { get; set; } = 1024;

    /// <summary>Vocos's feed-forward width (4096).</summary>
    public int VocoderIntermediate { get; set; } = 4096;

    /// <summary>Vocos's blocks (30).</summary>
    public int VocoderLayers { get; set; } = 30;

    // ---------------------------------------------------------------- training

    /// <summary>The network <see cref="Vevo{T}"/>'s training updates.</summary>
    public VevoStage CurrentStage { get; set; } = VevoStage.TextToContentStyle;

    /// <summary>The flow's learning rate (7.5e-5); the autoregressive models use <see cref="TtsModelOptions.LearningRate"/>.</summary>
    public double FlowLearningRate { get; set; } = 7.5e-5;

    /// <summary>Warm-up steps of the inverse-square-root schedule (32,000).</summary>
    public int WarmupSteps { get; set; } = 32000;

    /// <summary>AdamW's β1 (0.9).</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>AdamW's β2 (0.999).</summary>
    public double Beta2 { get; set; } = 0.999;

    /// <summary>AdamW's ε (1e-8).</summary>
    public double Epsilon { get; set; } = 1e-8;

    /// <summary>The seed of training's draws.</summary>
    public int TrainingSeed { get; set; } = 114;
}
