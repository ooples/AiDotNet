namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>Step-Audio's training stages (§4.2 and §5).</summary>
public enum StepAudioTrainingStage
{
    /// <summary>Pre-training stage 1: audio continuation (and text), the 5,120 new audio tokens' embedding and head rows
    /// at 5× the backbone's 2e-5.</summary>
    PreTraining1,

    /// <summary>Stage 2: audio continuation and audio-text interleaved data, the same rates.</summary>
    PreTraining2,

    /// <summary>Stage 3: continuation, interleaving, ASR and TTS data; every rate on one cosine 2e-5 → 5e-6.</summary>
    PreTraining3,

    /// <summary>The TTS model's supervised fine-tuning (§5.1.2): the two-turn chat format, one epoch, cosine 2e-5 → 2e-6.</summary>
    TtsFineTuning,

    /// <summary>AQTA supervised fine-tuning (§5.2.2): audio question, text answer, cosine 5.656e-5 → 5.656e-6.</summary>
    ChatFineTuning,

    /// <summary>The reward model (§5.2.4): Bradley–Terry on preference pairs, cosine 1.24e-5 → 6e-6.</summary>
    RewardModel,

    /// <summary>PPO against the reward model (§5.2.6): clip 0.2, KL 0.05, the critic warmed up 80 steps, cosine 1e-6 → 2e-7.</summary>
    Ppo,
}

/// <summary>Options for <see cref="StepAudio{T}"/>.</summary>
/// <remarks>
/// <para>The defaults are the release (stepfun-ai Step-Audio-Chat, Step-Audio-TTS-3B, Step-Audio-Tokenizer) as the paper
/// describes it: the dual-codebook tokenizer — the streaming Paraformer-large encoder's 16.7 Hz frames quantized to 1,024
/// k-means codes, and CosyVoice's S3 tokenizer at 25 Hz with 4,096 codes, interleaved 2:3 — the 130B Step-1 chat model,
/// and the 3B Step-1 TTS model whose dual-codebook tokens CosyVoice-300M-25Hz's flow and HiFT turn into 22.05 kHz
/// speech, with a separate music decoder for RAP and humming.</para>
/// <para><b>For Beginners:</b> Step-Audio hears speech as two kinds of tokens (what is said and how it sounds), answers in
/// text with a large language model, and speaks the answer with a smaller one.</para>
/// </remarks>
public class StepAudioOptions : TtsModelOptions
{
    /// <summary>Creates the release's Step-Audio.</summary>
    public StepAudioOptions()
    {
        SampleRate = 22050;
        HiddenDim = 12288;
        NumHeads = 96;
        VocabSize = 74752;
        MaxTextLength = 2048;
        DropoutRate = 0.0;
        LearningRate = 2e-5;
        WeightDecay = 0.1;
        Seed = 42;
    }

    // ---------------------------------------------------------------- the chat model (Step-1 130B)

    /// <summary>The chat model's key-value groups (8), layers (88) and feed-forward width (31,232); <c>HiddenDim</c> and
    /// <c>NumHeads</c> are its width (12,288) and heads (96). Null-sized (0 layers) leaves the chat model out.</summary>
    public int ChatGroups { get; set; } = 8;

    /// <inheritdoc cref="ChatGroups"/>
    public int ChatLayers { get; set; } = 88;

    /// <inheritdoc cref="ChatGroups"/>
    public int ChatIntermediate { get; set; } = 31232;

    // ---------------------------------------------------------------- the TTS model (Step-1 3B)

    /// <summary>The TTS model: width (3072), heads (48), key-value groups (4), layers (32), feed-forward (8192).</summary>
    public int TtsWidth { get; set; } = 3072;

    /// <inheritdoc cref="TtsWidth"/>
    public int TtsHeads { get; set; } = 48;

    /// <inheritdoc cref="TtsWidth"/>
    public int TtsGroups { get; set; } = 4;

    /// <inheritdoc cref="TtsWidth"/>
    public int TtsLayers { get; set; } = 32;

    /// <inheritdoc cref="TtsWidth"/>
    public int TtsIntermediate { get; set; } = 8192;

    /// <summary>Step-1's RMSNorm epsilon (1e-5).</summary>
    public double RmsEpsilon { get; set; } = 1e-5;

    /// <summary>Step-1's tokenizer (<c>tokenizer.model</c>); text needs it.</summary>
    public string? TokenizerPath { get; set; }

    // ---------------------------------------------------------------- the dual-codebook tokenizer

    /// <summary>The linguistic tokenizer's Paraformer encoder: filterbank bands (80), width (512), heads (4),
    /// feed-forward (2048), blocks (50); and its k-means codes (1,024).</summary>
    public int ParaformerMels { get; set; } = 80;

    /// <inheritdoc cref="ParaformerMels"/>
    public int ParaformerWidth { get; set; } = 512;

    /// <inheritdoc cref="ParaformerMels"/>
    public int ParaformerHeads { get; set; } = 4;

    /// <inheritdoc cref="ParaformerMels"/>
    public int ParaformerFeedForward { get; set; } = 2048;

    /// <inheritdoc cref="ParaformerMels"/>
    public int ParaformerBlocks { get; set; } = 50;

    /// <inheritdoc cref="ParaformerMels"/>
    public int LinguisticCodes { get; set; } = 1024;

    /// <summary>The filterbanks' Kaldi dither (the release's 1.0; 0 makes the tokens deterministic).</summary>
    public double LinguisticDither { get; set; } = 1.0;

    /// <summary>The semantic tokenizer (CosyVoice's speech_tokenizer_v1): width (1280), heads (20), layers (6), codes
    /// (4,096).</summary>
    public int SemanticWidth { get; set; } = 1280;

    /// <inheritdoc cref="SemanticWidth"/>
    public int SemanticHeads { get; set; } = 20;

    /// <inheritdoc cref="SemanticWidth"/>
    public int SemanticLayers { get; set; } = 6;

    /// <inheritdoc cref="SemanticWidth"/>
    public int SemanticCodes { get; set; } = 4096;

    // ---------------------------------------------------------------- the decoder (CosyVoice-300M-25Hz)

    /// <summary>The flow: width (512), heads (8), feed-forward (2048), encoder blocks (6), decoder channels (256, 256),
    /// decoder heads (8) of 64, blocks per level (4), middle blocks (12), Euler steps (10), mel bands (80).</summary>
    public int FlowWidth { get; set; } = 512;

    /// <inheritdoc cref="FlowWidth"/>
    public int FlowHeads { get; set; } = 8;

    /// <inheritdoc cref="FlowWidth"/>
    public int FlowFeedForward { get; set; } = 2048;

    /// <inheritdoc cref="FlowWidth"/>
    public int FlowEncoderLayers { get; set; } = 6;

    /// <inheritdoc cref="FlowWidth"/>
    public int[] DecoderChannels { get; set; } = { 256, 256 };

    /// <inheritdoc cref="FlowWidth"/>
    public int DecoderHeads { get; set; } = 8;

    /// <inheritdoc cref="FlowWidth"/>
    public int DecoderHeadDim { get; set; } = 64;

    /// <inheritdoc cref="FlowWidth"/>
    public int DecoderBlocks { get; set; } = 4;

    /// <inheritdoc cref="FlowWidth"/>
    public int DecoderMidBlocks { get; set; } = 12;

    /// <inheritdoc cref="FlowWidth"/>
    public int FlowTimesteps { get; set; } = 10;

    /// <inheritdoc cref="FlowWidth"/>
    public int MelChannelCount { get; set; } = 80;

    /// <summary>HiFT: base channels (512), harmonics (8), upsampling (8, 8) with kernels (16, 16), source kernels (7, 11).</summary>
    public int HiftBaseChannels { get; set; } = 512;

    /// <inheritdoc cref="HiftBaseChannels"/>
    public int HiftHarmonics { get; set; } = 8;

    /// <inheritdoc cref="HiftBaseChannels"/>
    public int[] HiftUpsampleRates { get; set; } = { 8, 8 };

    /// <inheritdoc cref="HiftBaseChannels"/>
    public int[] HiftUpsampleKernels { get; set; } = { 16, 16 };

    /// <inheritdoc cref="HiftBaseChannels"/>
    public int[] HiftSourceKernels { get; set; } = { 7, 11 };

    /// <summary>CAM++'s x-vector width (192), initial channels (128) and growth (32).</summary>
    public int SpeakerEmbeddingDimension { get; set; } = 192;

    /// <inheritdoc cref="SpeakerEmbeddingDimension"/>
    public int SpeakerEncoderChannels { get; set; } = 128;

    /// <inheritdoc cref="SpeakerEmbeddingDimension"/>
    public int SpeakerEncoderGrowthRate { get; set; } = 32;

    /// <summary>Whether to build the music decoder (CosyVoice-300M-25Hz-Music) the release uses for RAP and humming.</summary>
    public bool IncludeMusicDecoder { get; set; } = true;

    // ---------------------------------------------------------------- generation

    /// <summary>The chat model's sampling (stepaudio.py: temperature 0.7, top-p 0.9, at most 2048 new tokens).</summary>
    public double ChatTemperature { get; set; } = 0.7;

    /// <inheritdoc cref="ChatTemperature"/>
    public double ChatTopP { get; set; } = 0.9;

    /// <inheritdoc cref="ChatTemperature"/>
    public int ChatMaxNewTokens { get; set; } = 2048;

    /// <summary>The TTS model's sampling (tts.py: temperature 0.7, the repetition-aware processor's window 10 and ratio
    /// 0.1, at most 8192 tokens in all).</summary>
    public double TtsTemperature { get; set; } = 0.7;

    /// <inheritdoc cref="TtsTemperature"/>
    public int RepetitionWindow { get; set; } = 10;

    /// <inheritdoc cref="TtsTemperature"/>
    public double RepetitionRatio { get; set; } = 0.1;

    /// <inheritdoc cref="TtsTemperature"/>
    public int TtsMaxLength { get; set; } = 8192;

    /// <summary>A cap on the TTS model's new tokens on top of <see cref="TtsMaxLength"/> (the release has none).</summary>
    public int TtsMaxNewTokens { get; set; } = int.MaxValue;

    // ---------------------------------------------------------------- training

    /// <summary>The stage a training step follows. Each stage's rates are the paper's, scaled by <c>LearningRate</c> /
    /// 2e-5 (the paper's base rate and the default).</summary>
    public StepAudioTrainingStage Stage { get; set; } = StepAudioTrainingStage.PreTraining1;

    /// <summary>Steps of the stage's cosine schedule (not stated; 10,000) and its warmup (not stated; 0).</summary>
    public int TotalSteps { get; set; } = 10_000;

    /// <inheritdoc cref="TotalSteps"/>
    public int WarmupSteps { get; set; }

    /// <summary>Whether to build what RLHF needs (§5.2.3–5.2.6): a reward model and a critic (Step-1 chat-sized backbones
    /// with a scalar head) and the frozen reference policy for the KL penalty.</summary>
    public bool IncludeRlhfModels { get; set; }

    /// <summary>PPO: the clip (0.2), the KL penalty (0.05), the critic's warmup steps (80), GAE's γ and λ (not stated: 1
    /// and 0.95), and the sampled response length (not stated: 256).</summary>
    public double PpoClip { get; set; } = 0.2;

    /// <inheritdoc cref="PpoClip"/>
    public double KlCoefficient { get; set; } = 0.05;

    /// <inheritdoc cref="PpoClip"/>
    public int CriticWarmupSteps { get; set; } = 80;

    /// <inheritdoc cref="PpoClip"/>
    public double Gamma { get; set; } = 1.0;

    /// <inheritdoc cref="PpoClip"/>
    public double Lambda { get; set; } = 0.95;

    /// <inheritdoc cref="PpoClip"/>
    public int PpoResponseLength { get; set; } = 256;

    /// <summary>Gradient clipping (not stated: 1.0).</summary>
    public double MaxGradientNorm { get; set; } = 1.0;
}
