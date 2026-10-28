namespace AiDotNet.TextToSpeech.Latest;

/// <summary>Which of IndexTTS2's networks <see cref="IndexTTS2{T}"/> trains.</summary>
public enum IndexTts2Stage
{
    /// <summary>Text-to-semantic stage 1: the whole model on <c>[c, p, BT, text, BA, sem]</c>, the duration token
    /// zeroed with <see cref="IndexTTS2Options.DurationDropProbability"/>.</summary>
    TextToSemantic,

    /// <summary>Text-to-semantic stage 2: <c>[c + e, …]</c> with the speaker conditioner frozen, the emotion conditioner
    /// trained and a speaker classifier behind a gradient reversal layer pushing speaker identity out of e (Eq. 1).</summary>
    Emotion,

    /// <summary>Text-to-semantic stage 3: every conditioner frozen, the rest fine-tuned.</summary>
    FineTune,

    /// <summary>The semantic-to-mel flow matching (Eq. 2), with the GPT latents fused into half the samples.</summary>
    SemanticToMel,
}

/// <summary>How IndexTTS2 turns an emotion distribution into the emotion vector.</summary>
public enum IndexTts2EmotionMode
{
    /// <summary>The paper's Eq. 5: <c>Σ_e p_e · mean(V_e)</c> over each emotion's embedding set.</summary>
    Paper,

    /// <summary>The release's: per emotion, the row of its set whose speaker embedding is nearest (cosine) the voice's
    /// CAM++ style.</summary>
    ReleaseNearestSpeaker,

    /// <summary>The release's <c>use_random</c>: a random row per emotion.</summary>
    ReleaseRandomRow,
}

/// <summary>Options for <see cref="IndexTTS2{T}"/>.</summary>
/// <remarks>
/// <para>The paper (Zhou et al. 2025) fixes the design but few sizes; the defaults are the release's
/// (<c>IndexTeam/IndexTTS-2</c> <c>config.yaml</c>) except where the paper decides otherwise: the duration token
/// <c>p = W_sem h(T)</c> (<see cref="Duration"/>), the 7-emotion set and Eq. 5's mean (<see cref="EmotionMode"/>,
/// <see cref="EmotionLabels"/>), and a Qwen3-1.7B text-to-emotion model. <see cref="Release"/> gives the released
/// configuration (speed tokens, 8 emotions with nearest-speaker rows, Qwen3-0.6B), which the released weights need.</para>
/// <para><b>For Beginners:</b> IndexTTS2 clones a voice from a short recording and can set the speech's emotion and
/// length. These settings size its networks and control generation; the defaults follow the paper, and
/// <see cref="Release"/> matches the downloadable model.</para>
/// </remarks>
public class IndexTTS2Options : TtsModelOptions
{
    /// <summary>Creates the paper's configuration with the release's sizes.</summary>
    public IndexTTS2Options()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        FftSize = 1024;
        HiddenDim = 1280;
        NumHeads = 20;
        VocabSize = 12000;
        MaxTextLength = 600;
        DropoutRate = 0.1;
        LearningRate = 2e-4;
        WeightDecay = 0.01;
    }

    /// <summary>The released model's configuration: the speed tokens, the release's 8 emotions with nearest-speaker rows
    /// and group sizes, and Qwen3-0.6B for text-to-emotion.</summary>
    public static IndexTTS2Options Release() => new()
    {
        Duration = IndexTts2DurationConditioning.ReleaseSpeedTokens,
        EmotionMode = IndexTts2EmotionMode.ReleaseNearestSpeaker,
        EmotionLabels = new[] { "happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm" },
        EmotionGroupSizes = new[] { 3, 17, 2, 8, 4, 5, 10, 24 },
        EmotionHidden = 1024,
        EmotionIntermediate = 3072,
    };

    // ---------------------------------------------------------------- text

    /// <summary>The release's SentencePiece model (<c>bpe.model</c>), which turning text into tokens needs.</summary>
    public string? BpeModelPath { get; set; }

    /// <summary>Text tokens per synthesized segment (<c>max_text_tokens_per_segment</c>, 120).</summary>
    public int SegmentTokens { get; set; } = 120;

    /// <summary>Silence between segments, in milliseconds (<c>interval_silence</c>, 200).</summary>
    public int SegmentSilenceMilliseconds { get; set; } = 200;

    // ---------------------------------------------------------------- text-to-semantic

    /// <summary>GPT-2 layers (24).</summary>
    public int Layers { get; set; } = 24;

    /// <summary>Semantic positions (<c>max_mel_tokens</c>, 1815).</summary>
    public int MaxMelTokens { get; set; } = 1815;

    /// <summary>Semantic codes plus the start and stop tokens (8194).</summary>
    public int MelCodes { get; set; } = 8194;

    /// <summary>The speaker perceiver's latents (32).</summary>
    public int ConditionLatents { get; set; } = 32;

    /// <summary>The speaker conformer's width (512).</summary>
    public int ConditionDim { get; set; } = 512;

    /// <summary>The speaker conformer's feed-forward width (2048).</summary>
    public int ConditionLinearUnits { get; set; } = 2048;

    /// <summary>The speaker conformer's heads (8).</summary>
    public int ConditionHeads { get; set; } = 8;

    /// <summary>The speaker conformer's blocks (6).</summary>
    public int ConditionBlocks { get; set; } = 6;

    /// <summary>The emotion conformer's width (512).</summary>
    public int EmotionConditionDim { get; set; } = 512;

    /// <summary>The emotion conformer's feed-forward width (1024).</summary>
    public int EmotionConditionLinearUnits { get; set; } = 1024;

    /// <summary>The emotion conformer's heads (4).</summary>
    public int EmotionConditionHeads { get; set; } = 4;

    /// <summary>The emotion conformer's blocks (4).</summary>
    public int EmotionConditionBlocks { get; set; } = 4;

    /// <summary>The perceivers' feed-forward multiple (2).</summary>
    public int PerceiverMultiple { get; set; } = 2;

    /// <summary>How the target length reaches the model: the paper's duration token (default) or the release's speed
    /// tokens.</summary>
    public IndexTts2DurationConditioning Duration { get; set; } = IndexTts2DurationConditioning.Paper;

    /// <summary>The duration token's drop probability in stage 1 (§ Training: 30%).</summary>
    public double DurationDropProbability { get; set; } = 0.3;

    /// <summary>The number of semantic tokens to generate per segment (the paper's duration control); null generates
    /// freely. With <see cref="IndexTts2DurationConditioning.ReleaseSpeedTokens"/> it is ignored.</summary>
    public int? TargetSemanticTokens { get; set; }

    // ---------------------------------------------------------------- decoding

    /// <summary>Whether the beam candidates are sampled (<c>do_sample</c>, true) rather than taken best-first.</summary>
    public bool Sample { get; set; } = true;

    /// <summary>Beams (<c>num_beams</c>, 3).</summary>
    public int Beams { get; set; } = 3;

    /// <summary>Top-k (30).</summary>
    public int TopK { get; set; } = 30;

    /// <summary>Top-p (0.8).</summary>
    public double TopP { get; set; } = 0.8;

    /// <summary>Temperature (0.8).</summary>
    public double Temperature { get; set; } = 0.8;

    /// <summary>Repetition penalty (10).</summary>
    public double RepetitionPenalty { get; set; } = 10.0;

    /// <summary>The most semantic tokens a segment generates (<c>max_mel_tokens</c> at inference, 1500).</summary>
    public int MaxGeneratedTokens { get; set; } = 1500;

    /// <summary>The seed of generation's draws and the flow's noise.</summary>
    public int SamplingSeed { get; set; } = 1234;

    // ---------------------------------------------------------------- semantic features

    /// <summary>The rate the semantic model reads (16 kHz).</summary>
    public int SemanticSampleRate { get; set; } = 16000;

    /// <summary>w2v-BERT 2.0's width (1024).</summary>
    public int SemanticHiddenSize { get; set; } = 1024;

    /// <summary>w2v-BERT 2.0's heads (16).</summary>
    public int SemanticHeads { get; set; } = 16;

    /// <summary>w2v-BERT 2.0's feed-forward width (4096).</summary>
    public int SemanticIntermediateSize { get; set; } = 4096;

    /// <summary>The hidden state the features come from (17).</summary>
    public int SemanticLayer { get; set; } = 17;

    /// <summary>The semantic codec's Vocos width (384).</summary>
    public int SemanticCodecDim { get; set; } = 384;

    /// <summary>The semantic codec's Vocos feed-forward width (2048).</summary>
    public int SemanticCodecIntermediate { get; set; } = 2048;

    /// <summary>The semantic codec's Vocos blocks (12).</summary>
    public int SemanticCodecLayers { get; set; } = 12;

    /// <summary>Semantic codes (8192).</summary>
    public int SemanticCodebookSize { get; set; } = 8192;

    /// <summary>The semantic codebook's dimension (8).</summary>
    public int SemanticCodebookDim { get; set; } = 8;

    /// <summary>The longest prompt the release reads, in seconds (15).</summary>
    public double MaxPromptSeconds { get; set; } = 15;

    // ---------------------------------------------------------------- semantic-to-mel

    /// <summary>The DiT's width (512).</summary>
    public int DitHidden { get; set; } = 512;

    /// <summary>The DiT's heads (8).</summary>
    public int DitHeads { get; set; } = 8;

    /// <summary>The DiT's blocks (13).</summary>
    public int DitDepth { get; set; } = 13;

    /// <summary>The CAM++ style's width (192).</summary>
    public int StyleDim { get; set; } = 192;

    /// <summary>The length regulator's width (512).</summary>
    public int RegulatorChannels { get; set; } = 512;

    /// <summary>The length regulator's convolution stages (4).</summary>
    public int RegulatorStages { get; set; } = 4;

    /// <summary>The WaveNet head's width (512).</summary>
    public int WavenetHidden { get; set; } = 512;

    /// <summary>The WaveNet head's layers (8).</summary>
    public int WavenetLayers { get; set; } = 8;

    /// <summary>The WaveNet head's kernel (5).</summary>
    public int WavenetKernel { get; set; } = 5;

    /// <summary>The WaveNet head's dropout (0.2).</summary>
    public double WavenetDropout { get; set; } = 0.2;

    /// <summary>The conditioning dropout of flow-matching training (0.1).</summary>
    public double ClassDropout { get; set; } = 0.1;

    /// <summary>The GPT latent fusion probability in training (§ S2M: 50%).</summary>
    public double LatentFusionProbability { get; set; } = 0.5;

    /// <summary>Euler steps (25).</summary>
    public int DiffusionSteps { get; set; } = 25;

    /// <summary>Classifier-free guidance (<c>inference_cfg_rate</c>, 0.7).</summary>
    public double Guidance { get; set; } = 0.7;

    /// <summary>Mel frames per semantic token (1.72: 22050 / 256 frames a second over 50 tokens).</summary>
    public double FramesPerToken { get; set; } = 1.72;

    /// <summary>CAM++'s growth rate (32).</summary>
    public int SpeakerEncoderGrowthRate { get; set; } = 32;

    /// <summary>CAM++'s TDNN channels (128).</summary>
    public int SpeakerEncoderChannels { get; set; } = 128;

    // ---------------------------------------------------------------- vocoder (BigVGAN-v2, 22 kHz, 80 bands, 256×)

    /// <summary>BigVGAN's first width (1536).</summary>
    public int VocoderInitialChannels { get; set; } = 1536;

    /// <summary>BigVGAN's upsampling rates (4, 4, 2, 2, 2, 2).</summary>
    public int[] VocoderUpsampleRates { get; set; } = { 4, 4, 2, 2, 2, 2 };

    /// <summary>BigVGAN's upsampling kernels (8, 8, 4, 4, 4, 4).</summary>
    public int[] VocoderUpsampleKernels { get; set; } = { 8, 8, 4, 4, 4, 4 };

    /// <summary>BigVGAN's AMP block kernels (3, 7, 11).</summary>
    public int[] VocoderResBlockKernels { get; set; } = { 3, 7, 11 };

    /// <summary>BigVGAN's AMP block dilations ((1, 3, 5) each).</summary>
    public int[][] VocoderResBlockDilations { get; set; } = { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } };

    // ---------------------------------------------------------------- emotion

    /// <summary>How an emotion distribution becomes the emotion vector (the paper's Eq. 5 by default).</summary>
    public IndexTts2EmotionMode EmotionMode { get; set; } = IndexTts2EmotionMode.Paper;

    /// <summary>The emotions, in the order of the emotion table's groups and of an emotion vector's weights (the paper's
    /// seven by default).</summary>
    public string[] EmotionLabels { get; set; } = { "anger", "happiness", "fear", "disgust", "sadness", "surprise", "neutral" };

    /// <summary>The emotion table's rows per emotion (the release's <c>emo_num</c>); null when the table is not loaded.</summary>
    public int[]? EmotionGroupSizes { get; set; }

    /// <summary>The text-to-emotion model directory (a Qwen3 <c>model.safetensors</c> and <c>tokenizer.json</c>).</summary>
    public string? EmotionModelPath { get; set; }

    /// <summary>The text-to-emotion model's width (Qwen3-1.7B: 2048).</summary>
    public int EmotionHidden { get; set; } = 2048;

    /// <summary>The text-to-emotion model's layers (28).</summary>
    public int EmotionLayers { get; set; } = 28;

    /// <summary>The text-to-emotion model's heads (16).</summary>
    public int EmotionHeads { get; set; } = 16;

    /// <summary>The text-to-emotion model's key-value heads (8).</summary>
    public int EmotionKeyValueHeads { get; set; } = 8;

    /// <summary>The text-to-emotion model's feed-forward width (Qwen3-1.7B: 6144).</summary>
    public int EmotionIntermediate { get; set; } = 6144;

    /// <summary>The text-to-emotion model's head width (128).</summary>
    public int EmotionHeadDim { get; set; } = 128;

    /// <summary>The text-to-emotion model's vocabulary (151,936).</summary>
    public int EmotionVocabulary { get; set; } = 151936;

    /// <summary>The LoRA rank the text-to-emotion distillation trains (§ T2E; 8).</summary>
    public int EmotionLoraRank { get; set; } = 8;

    /// <summary>The most tokens the text-to-emotion model generates (256).</summary>
    public int EmotionMaxTokens { get; set; } = 256;

    // ---------------------------------------------------------------- training

    /// <summary>The stage <see cref="IndexTTS2{T}"/> trains.</summary>
    public IndexTts2Stage CurrentStage { get; set; } = IndexTts2Stage.TextToSemantic;

    /// <summary>The speaker classifier's speakers (stage 2).</summary>
    public int SpeakerCount { get; set; } = 1;

    /// <summary>The speaker classifier's loss coefficient α in Eq. 1 (the gradient reversal scale; the paper gives no
    /// value).</summary>
    public double SpeakerAdversaryWeight { get; set; } = 1.0;

    /// <summary>AdamW β₁.</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>AdamW β₂.</summary>
    public double Beta2 { get; set; } = 0.999;

    /// <summary>AdamW ε.</summary>
    public double Epsilon { get; set; } = 1e-8;

    /// <summary>The seed of the networks' initialization and of training's draws.</summary>
    public int TrainingSeed { get; set; } = 42;
}
