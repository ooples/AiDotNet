using AiDotNet.TextToSpeech.FrontEnd;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The GPT-SoVITS release a <see cref="GPTSoVITS{T}"/> follows (each has its own released weights).</summary>
public enum GptSoVitsVersion
{
    /// <summary>v1: Chinese, Japanese and English with v1's 322 symbols and pypinyin; the GAN synthesizer.</summary>
    V1,

    /// <summary>v2: the 732-symbol set adding Korean and Cantonese, G2PW for Chinese; the GAN synthesizer.</summary>
    V2,

    /// <summary>v2Pro: v2 with an ERes2NetV2 speaker embedding added to the style vector.</summary>
    V2Pro,

    /// <summary>v2ProPlus: v2Pro with a wider generator (the default).</summary>
    V2ProPlus,

    /// <summary>v3: the text encoder's hidden state to 24 kHz mels by conditional flow matching, then BigVGAN.</summary>
    V3,

    /// <summary>v4: flow matching to 32 kHz-hop mels, then a 48 kHz HiFi-GAN vocoder.</summary>
    V4,

    /// <summary>V5 dev: v4's design without the step embedding, with classifier-free guidance and rolling-tail chunks.</summary>
    V5Dev,

    /// <summary>V5 turbo: V5 sampled in four steps without guidance.</summary>
    V5Turbo,
}

/// <summary>Which of GPT-SoVITS's networks <see cref="GPTSoVITS{T}"/> trains.</summary>
public enum GptSoVitsStage
{
    /// <summary>s1: the text-to-semantic GPT (s1_train.py; ScaledAdam).</summary>
    TextToSemantic,

    /// <summary>s2 for v1 to v2ProPlus: the SoVITS synthesizer and its discriminators as a GAN (s2_train.py).</summary>
    SoVits,

    /// <summary>s2 for v3 to V5: the flow-matching synthesizer, by default LoRA on the DiT's attention
    /// (s2_train_v3_lora.py; <see cref="GPTSoVITSOptions.LoraRank"/> 0 trains it all, as s2_train_v3.py).</summary>
    FlowMatching,
}

/// <summary>Options for <see cref="GPTSoVITS{T}"/>.</summary>
/// <remarks>
/// <para>GPT-SoVITS has no paper; the release (RVC-Boss/GPT-SoVITS) defines it. The defaults are its v2ProPlus
/// configuration (<c>s1longer-v2.yaml</c>, <c>s2v2ProPlus.json</c>) and the inference interface's defaults. The sizes
/// marked "by version" follow <see cref="Version"/> unless set.</para>
/// <para><b>For Beginners:</b> GPT-SoVITS copies a voice from a few seconds of recording: a GPT writes "semantic tokens"
/// for the text, and a second network turns them into speech in the reference voice. Pick a <see cref="Version"/> to
/// match the weights you load; the rest can usually stay as it is.</para>
/// </remarks>
public class GPTSoVITSOptions : TtsModelOptions
{
    /// <summary>Creates v2ProPlus's configuration.</summary>
    public GPTSoVITSOptions()
    {
        SampleRate = 32000;
        MelChannels = 128;
        HopSize = 640;
        FftSize = 2048;
        HiddenDim = 512;
        NumHeads = 16;
        MaxTextLength = 1000;
        DropoutRate = 0.0;
        LearningRate = 0.01;
        WeightDecay = 0.01;
    }

    /// <summary>The release followed (v2ProPlus).</summary>
    public GptSoVitsVersion Version { get; set; } = GptSoVitsVersion.V2ProPlus;

    /// <summary>The network <see cref="GPTSoVITS{T}"/>'s training updates.</summary>
    public GptSoVitsStage CurrentStage { get; set; } = GptSoVitsStage.TextToSemantic;

    // ---------------------------------------------------------------- text

    /// <summary>How the target text's language is read (the interface's default: Chinese).</summary>
    public GptSoVitsTextLanguage TextLanguage { get; set; } = GptSoVitsTextLanguage.Chinese;

    /// <summary>How the voice's prompt text is read (the interface's default: Chinese).</summary>
    public GptSoVitsTextLanguage PromptLanguage { get; set; } = GptSoVitsTextLanguage.Chinese;

    /// <summary>How the target text is cut into sentences (the interface's default: every four sentences).</summary>
    public GptSoVitsTextCut Cut { get; set; } = GptSoVitsTextCut.FourSentences;

    /// <summary>fastText's language identifier (<c>lid.176.bin</c>), which every language setting but English needs.</summary>
    public string? LanguageModelPath { get; set; }

    /// <summary>Open JTalk's dictionary directory (open_jtalk_dic_utf_8-1.11), which Japanese needs.</summary>
    public string? OpenJTalkDictionaryPath { get; set; }

    /// <summary>The G2PW model directory (G2PWModel_1.1 with <c>g2pW.onnx</c>), which Chinese after v1 needs.</summary>
    public string? G2pwDirectory { get; set; }

    // ---------------------------------------------------------------- text-to-semantic (s1)

    /// <summary>The GPT's layers (24); its width is <see cref="TtsModelOptions.HiddenDim"/> (512) and its heads
    /// <see cref="TtsModelOptions.NumHeads"/> (16).</summary>
    public int T2SLayers { get; set; } = 24;

    /// <summary>Top-k of semantic sampling (the interface's 15).</summary>
    public int TopK { get; set; } = 15;

    /// <summary>Top-p of semantic sampling (the interface's 1).</summary>
    public double TopP { get; set; } = 1.0;

    /// <summary>The sampling temperature (the interface's 1).</summary>
    public double Temperature { get; set; } = 1.0;

    /// <summary>The repetition penalty (1.35).</summary>
    public double RepetitionPenalty { get; set; } = 1.35;

    /// <summary>The longest generation in seconds of 50 Hz semantic tokens (<c>max_sec</c>, 54).</summary>
    public int MaxSeconds { get; set; } = 54;

    /// <summary>The seed of semantic sampling and of the synthesizer's noise.</summary>
    public int SamplingSeed { get; set; } = 1234;

    // ---------------------------------------------------------------- SoVITS synthesizer

    /// <summary>The prior's channels (192).</summary>
    public int InterChannels { get; set; } = 192;

    /// <summary>The text encoder's width (192).</summary>
    public int SynthesizerHidden { get; set; } = 192;

    /// <summary>The text encoder's feed-forward width (768).</summary>
    public int SynthesizerFilter { get; set; } = 768;

    /// <summary>The text encoder's heads (2).</summary>
    public int SynthesizerHeads { get; set; } = 2;

    /// <summary>The text encoder's layers (6; its SSL and third Transformers have half).</summary>
    public int SynthesizerLayers { get; set; } = 6;

    /// <summary>The style vector's width (by version: 512 for v1, v2 and v3 to V5, 1024 for v2Pro and v2ProPlus).</summary>
    public int? StyleChannels { get; set; }

    /// <summary>MRTE's width (512).</summary>
    public int MrteChannels { get; set; } = 512;

    /// <summary>The style encoder's hidden width (128).</summary>
    public int StyleHidden { get; set; } = 128;

    /// <summary>The synthesizer's dropout (by version: 0.1 from s2.json, 0 for v2Pro and v2ProPlus).</summary>
    public double? SynthesizerDropout { get; set; }

    /// <summary>The GAN generator's upsampling rates (10, 8, 2, 2, 2: 640 samples per frame).</summary>
    public int[] UpsampleRates { get; set; } = { 10, 8, 2, 2, 2 };

    /// <summary>The GAN generator's transposed-convolution kernels (by version: 16, 16, 8, 2, 2; 20, 16, 8, 2, 2 for
    /// v2ProPlus).</summary>
    public int[]? UpsampleKernels { get; set; }

    /// <summary>The GAN generator's first width (by version: 512; 768 for v2ProPlus).</summary>
    public int? UpsampleInitialChannels { get; set; }

    /// <summary>The GAN generator's residual-block kernels (3, 7, 11).</summary>
    public int[] ResblockKernels { get; set; } = { 3, 7, 11 };

    /// <summary>The GAN generator's residual-block dilations (1, 3, 5 each).</summary>
    public int[][] ResblockDilations { get; set; } = { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } };

    /// <summary>The semantic codebook's size (1024; the GPT's vocabulary is this plus the end token).</summary>
    public int SemanticCodebookSize { get; set; } = 1024;

    // ---------------------------------------------------------------- flow matching (v3 to V5)

    /// <summary>The DiT's width (1024).</summary>
    public int DitDim { get; set; } = 1024;

    /// <summary>The DiT's blocks (22).</summary>
    public int DitDepth { get; set; } = 22;

    /// <summary>The DiT's heads (16).</summary>
    public int DitHeads { get; set; } = 16;

    /// <summary>The DiT's head width (64).</summary>
    public int DitHeadDim { get; set; } = 64;

    /// <summary>The DiT's ConvNeXt blocks over the features (4).</summary>
    public int DitConvLayers { get; set; } = 4;

    /// <summary>The bridge's and wns1's width (512).</summary>
    public int BridgeChannels { get; set; } = 512;

    /// <summary>Euler steps of flow matching (null: the release's 32 for v3 and v5dev, 8 for v4, 4 for v5turbo).</summary>
    public int? SampleSteps { get; set; }

    /// <summary>Classifier-free guidance (null: 1.3 for v5dev, none otherwise; only V5 reads it).</summary>
    public double? CfgRate { get; set; }

    // ---------------------------------------------------------------- vocoder (v3 to V5)

    /// <summary>The vocoder's first width (by version: BigVGAN's 1536 for v3; 512 for v4 and V5).</summary>
    public int? VocoderInitialChannels { get; set; }

    /// <summary>The vocoder's upsampling rates (by version: 4, 4, 2, 2, 2, 2 for v3; 10, 6, 2, 2, 2 after).</summary>
    public int[]? VocoderUpsampleRates { get; set; }

    /// <summary>The vocoder's upsampling kernels (by version: 8, 8, 4, 4, 4, 4 for v3; 20, 12, 4, 4, 4 after).</summary>
    public int[]? VocoderUpsampleKernels { get; set; }

    /// <summary>The vocoder's residual-block kernels (3, 7, 11).</summary>
    public int[] VocoderResblockKernels { get; set; } = { 3, 7, 11 };

    /// <summary>The vocoder's residual-block dilations (1, 3, 5 each).</summary>
    public int[][] VocoderResblockDilations { get; set; } = { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } };

    // ---------------------------------------------------------------- SSL, BERT and speaker models

    /// <summary>chinese-hubert-base's width (768; the SSL features the synthesizer quantizes).</summary>
    public int HubertHidden { get; set; } = 768;

    /// <summary>chinese-hubert-base's layers (12).</summary>
    public int HubertLayers { get; set; } = 12;

    /// <summary>chinese-hubert-base's heads (12).</summary>
    public int HubertHeads { get; set; } = 12;

    /// <summary>chinese-hubert-base's feed-forward width (3072).</summary>
    public int HubertIntermediate { get; set; } = 3072;

    /// <summary>The feature encoder's convolution widths (512 each).</summary>
    public int[] HubertConvDims { get; set; } = { 512, 512, 512, 512, 512, 512, 512 };

    /// <summary>The position convolution's kernel (128).</summary>
    public int HubertPositionKernel { get; set; } = 128;

    /// <summary>The position convolution's groups (16).</summary>
    public int HubertPositionGroups { get; set; } = 16;

    /// <summary>chinese-roberta-wwm-ext-large's width (1024; the GPT's BERT features).</summary>
    public int BertHidden { get; set; } = 1024;

    /// <summary>chinese-roberta-wwm-ext-large's layers (24; the features are hidden state −3).</summary>
    public int BertLayers { get; set; } = 24;

    /// <summary>chinese-roberta-wwm-ext-large's heads (16).</summary>
    public int BertHeads { get; set; } = 16;

    /// <summary>chinese-roberta-wwm-ext-large's feed-forward width (4096).</summary>
    public int BertIntermediate { get; set; } = 4096;

    /// <summary>ERes2NetV2's base width (64; v2Pro's 20,480-dimensional embedding is 8 · 4 · 10 times it).</summary>
    public int SpeakerChannels { get; set; } = 64;

    /// <summary>ERes2NetV2's blocks per stage (3, 4, 6, 3).</summary>
    public int[] SpeakerBlocks { get; set; } = { 3, 4, 6, 3 };

    // ---------------------------------------------------------------- inference

    /// <summary>The speaking rate (1).</summary>
    public double Speed { get; set; } = 1.0;

    /// <summary>Silence after each sentence, in seconds (0.3).</summary>
    public double PauseSeconds { get; set; } = 0.3;

    /// <summary>The prior's noise scale in the GAN synthesizer's decode (0.5).</summary>
    public double NoiseScale { get; set; } = 0.5;

    // ---------------------------------------------------------------- training

    /// <summary>s1's ScaledAdam learning rate before its first step (0.01; the release's schedule then holds 0.002).</summary>
    public double T2SLearningRate { get; set; } = 0.01;

    /// <summary>The rate s1's schedule locks to after the first step (0.002).</summary>
    public double T2SLockedLearningRate { get; set; } = 0.002;

    /// <summary>Batches s1 sums before each step (4: it steps at batch_idx &gt; 0 and batch_idx % 4 == 0).</summary>
    public int T2SAccumulation { get; set; } = 4;

    /// <summary>Whether s1 adds the reference-free DPO loss (<c>if_dpo</c>; off by default).</summary>
    public bool UseDpo { get; set; }

    /// <summary>s2's AdamW learning rate (1e-4).</summary>
    public double SynthesizerLearningRate { get; set; } = 1e-4;

    /// <summary>s2's AdamW β₁ (0.8).</summary>
    public double Beta1 { get; set; } = 0.8;

    /// <summary>s2's AdamW β₂ (0.99).</summary>
    public double Beta2 { get; set; } = 0.99;

    /// <summary>s2's AdamW ε (1e-9).</summary>
    public double Epsilon { get; set; } = 1e-9;

    /// <summary>s2's per-epoch learning-rate decay (0.999875).</summary>
    public double LearningRateDecay { get; set; } = 0.999875;

    /// <summary>Training steps per epoch, over which <see cref="LearningRateDecay"/> applies once (the dataset's batch
    /// count; null decays nothing).</summary>
    public int? UpdatesPerEpoch { get; set; }

    /// <summary>The rate factor of the text embedding, the text Transformer and MRTE in the GAN stage (0.4).</summary>
    public double TextLowLearningRate { get; set; } = 0.4;

    /// <summary>The mel loss's weight (45).</summary>
    public double MelLossWeight { get; set; } = 45;

    /// <summary>The KL loss's weight (1).</summary>
    public double KlLossWeight { get; set; } = 1;

    /// <summary>The generator's training segment in samples (20480: 32 frames).</summary>
    public int SegmentSize { get; set; } = 20480;

    /// <summary>The flow-matching stage's LoRA rank (32, the interface's default; 0 trains the whole synthesizer).</summary>
    public int LoraRank { get; set; } = 32;

    /// <summary>The training draws' seed.</summary>
    public int TrainingSeed { get; set; } = 1234;
}
