namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>Which audio streams Moshi's Depth Transformer predicts.</summary>
public enum MoshiDepthLayout
{
    /// <summary>The paper's (Eq. 6, 7): text, Moshi's Q audio streams and the user's Q, so the model also simulates the
    /// user and whole dialogues can be sampled.</summary>
    AllStreams,

    /// <summary>The release's (<c>dep_q = 8</c>, kyutai/moshiko and moshika): Moshi's Q audio streams; the user's are input
    /// only.</summary>
    MoshiStreams,
}

/// <summary>The delay between Moshi's text and audio streams (§3.4.4, App. C).</summary>
public enum MoshiVariant
{
    /// <summary>The dialogue model: text and audio aligned (delay 0), acoustic delay τ.</summary>
    Dialogue,

    /// <summary>The streaming TTS model: the audio <see cref="MoshiOptions.TextAudioDelaySeconds"/> behind the text, so the
    /// text dictates the speech.</summary>
    Tts,

    /// <summary>The streaming ASR model: the text <see cref="MoshiOptions.TextAudioDelaySeconds"/> behind the audio, so the
    /// audio dictates the text.</summary>
    Asr,
}

/// <summary>Moshi's training stages (§4.4, Table 1).</summary>
public enum MoshiTrainingStage
{
    /// <summary>Pre-training on unsupervised single-stream speech (text masked 30%, text delay ±0.6 s), half the batches
    /// text-only with their own optimizer state.</summary>
    PreTraining,

    /// <summary>Post-training on diarized speech split into two streams, text delay 0, text-only batches 10% of the time.</summary>
    PostTraining,

    /// <summary>Fine-tuning on Fisher's real two-speaker conversations.</summary>
    FisherFineTuning,

    /// <summary>Instruct fine-tuning on synthetic dialogues, the user's stream augmented.</summary>
    InstructFineTuning,
}

/// <summary>Options for <see cref="Moshi{T}"/> and its speech recognizer.</summary>
/// <remarks>
/// <para>The defaults are the paper's Moshi: the Helium 7B Temporal Transformer (32 layers, d 4096, 32 heads, feed-forward
/// 11264 behind a SiLU gate, RMSNorm, RoPE, a 3000-step context), the 6-layer, 1024-wide Depth Transformer with weights per
/// codebook, Mimi's 8 codebooks of 2048 at 12.5 Hz, the inner monologue's 32,000-token text stream, an acoustic delay of 1,
/// and the Depth Transformer predicting every stream (Eq. 6). <see cref="Release"/> is kyutai/moshiko.</para>
/// <para><b>For Beginners:</b> Moshi listens and speaks at the same time: every 80 ms it reads the user's sound, writes a
/// token of what it is about to say, and the sound of its own voice. It can talk over you, pause, and be interrupted.</para>
/// </remarks>
public class MoshiOptions : TtsModelOptions
{
    /// <summary>Creates the paper's Moshi.</summary>
    public MoshiOptions()
    {
        SampleRate = 24000;
        HiddenDim = 4096;
        NumHeads = 32;
        VocabSize = 32000;
        MaxTextLength = 3000;
        DropoutRate = 0.0;
        LearningRate = 3e-5;
        WeightDecay = 0.1;
        Seed = 42;
    }

    /// <summary>The released kyutai/moshiko-pytorch-bf16: the Depth Transformer predicts Moshi's streams only, behind a
    /// 2816-wide gate (feed-forward 4224).</summary>
    public static MoshiOptions Release() => new() { DepthLayout = MoshiDepthLayout.MoshiStreams, DepthFeedForward = 4224 };

    // ---------------------------------------------------------------- the RQ-Transformer

    /// <summary>The Temporal Transformer's layers (32), feed-forward width before the gate (4.125 · 4096 = 16896, a gate of
    /// 11264) and context (3000 steps, 4 min).</summary>
    public int Layers { get; set; } = 32;

    /// <inheritdoc cref="Layers"/>
    public int FeedForward { get; set; } = 16896;

    /// <inheritdoc cref="Layers"/>
    public int Context { get; set; } = 3000;

    /// <summary>The Depth Transformer's width (1024), layers (6), heads (16) and feed-forward width before the gate (6144,
    /// Table 1's MLP of 4096; the release's 4224 gives 2816).</summary>
    public int DepthDim { get; set; } = 1024;

    /// <inheritdoc cref="DepthDim"/>
    public int DepthLayers { get; set; } = 6;

    /// <inheritdoc cref="DepthDim"/>
    public int DepthHeads { get; set; } = 16;

    /// <inheritdoc cref="DepthDim"/>
    public int DepthFeedForward { get; set; } = 6144;

    /// <summary>Which streams the Depth Transformer predicts.</summary>
    public MoshiDepthLayout DepthLayout { get; set; } = MoshiDepthLayout.AllStreams;

    /// <summary>Mimi codebooks per speaker, Q (8), and their size (2048).</summary>
    public int Codebooks { get; set; } = 8;

    /// <inheritdoc cref="Codebooks"/>
    public int CodebookSize { get; set; } = 2048;

    /// <summary>The acoustic delay τ in steps (1; Table 1's pre-training used 2).</summary>
    public int AcousticDelay { get; set; } = 1;

    /// <summary>The text/audio delay mode, and its size for <see cref="MoshiVariant.Tts"/> and <see cref="MoshiVariant.Asr"/>
    /// (App. C: 2 s, 25 steps).</summary>
    public MoshiVariant Variant { get; set; } = MoshiVariant.Dialogue;

    /// <inheritdoc cref="Variant"/>
    public double TextAudioDelaySeconds { get; set; } = 2.0;

    /// <summary>The text stream's PAD (3) and EPAD (0) tokens.</summary>
    public int TextPadding { get; set; } = 3;

    /// <inheritdoc cref="TextPadding"/>
    public int TextEndPadding { get; set; } = 0;

    /// <summary>Helium's SentencePiece model (<c>tokenizer_spm_32k_3.model</c>); text needs it.</summary>
    public string? TokenizerPath { get; set; }

    // ---------------------------------------------------------------- Mimi

    /// <summary>Mimi's SEANet width (64), latent width (512), Transformer layers (8) and codebooks it holds (32, the first
    /// <see cref="Codebooks"/> used).</summary>
    public int MimiFilters { get; set; } = 64;

    /// <inheritdoc cref="MimiFilters"/>
    public int MimiDimension { get; set; } = 512;

    /// <inheritdoc cref="MimiFilters"/>
    public int MimiTransformerLayers { get; set; } = 8;

    /// <inheritdoc cref="MimiFilters"/>
    public int MimiCodebooks { get; set; } = 32;

    /// <summary>Mimi's SEANet ratios as the release lists them (8, 6, 5, 4), its Transformers' heads (8), feed-forward width
    /// (2048), context (250 frames) and quantizer width (256).</summary>
    public int[] MimiRatios { get; set; } = { 8, 6, 5, 4 };

    /// <inheritdoc cref="MimiRatios"/>
    public int MimiHeads { get; set; } = 8;

    /// <inheritdoc cref="MimiRatios"/>
    public int MimiFeedForward { get; set; } = 2048;

    /// <inheritdoc cref="MimiRatios"/>
    public int MimiContext { get; set; } = 250;

    /// <inheritdoc cref="MimiRatios"/>
    public int MimiCodebookDim { get; set; } = 256;

    /// <summary>Whether to build what training Mimi needs: the multi-scale STFT discriminator and WavLM for distillation.</summary>
    public bool IncludeMimiTraining { get; set; }

    /// <summary>Mimi's optimizer (§3.3): AdamW at 8e-4 with β (0.5, 0.9), weight decay 5e-2 on the Transformers only, and an
    /// exponential moving average of the weights at 0.99.</summary>
    public double MimiLearningRate { get; set; } = 8e-4;

    /// <inheritdoc cref="MimiLearningRate"/>
    public double MimiWeightDecay { get; set; } = 5e-2;

    /// <inheritdoc cref="MimiLearningRate"/>
    public double MimiWeightAverageDecay { get; set; } = 0.99;

    /// <summary>The probability of quantizing a training sequence (§3.3: 0.5), and whether the acoustic codebooks drop out.</summary>
    public double MimiQuantizationRate { get; set; } = 0.5;

    /// <inheritdoc cref="MimiQuantizationRate"/>
    public bool MimiQuantizerDropout { get; set; } = true;

    /// <summary>The adversarial-only losses' weights (Audiocraft's compression defaults the paper points to: adversarial 4,
    /// feature 4) under the loss balancer, and the WavLM distillation's (not stated; 1).</summary>
    public double MimiAdversarialWeight { get; set; } = 4.0;

    /// <inheritdoc cref="MimiAdversarialWeight"/>
    public double MimiFeatureWeight { get; set; } = 4.0;

    /// <inheritdoc cref="MimiAdversarialWeight"/>
    public double MimiDistillationWeight { get; set; } = 1.0;

    /// <summary>The multi-scale STFT discriminator's windows (Audiocraft's compression default: 1024, 2048, 512, 256, 128,
    /// hop a quarter) and filters (32).</summary>
    public int[] MimiDiscriminatorWindows { get; set; } = { 1024, 2048, 512, 256, 128 };

    /// <inheritdoc cref="MimiDiscriminatorWindows"/>
    public int MimiDiscriminatorFilters { get; set; } = 32;

    /// <summary>WavLM Large's width (1024), layers (24), heads (16) and feature-encoder channels (512) for the distillation
    /// targets.</summary>
    public int WavLmWidth { get; set; } = 1024;

    /// <inheritdoc cref="WavLmWidth"/>
    public int WavLmLayers { get; set; } = 24;

    /// <inheritdoc cref="WavLmWidth"/>
    public int WavLmHeads { get; set; } = 16;

    /// <inheritdoc cref="WavLmWidth"/>
    public int WavLmConvChannels { get; set; } = 512;

    // ---------------------------------------------------------------- generation

    /// <summary>Sampling (the release's LMGen): audio at temperature 0.8 with top-k 250, text at 0.7 with top-k 25.</summary>
    public double AudioTemperature { get; set; } = 0.8;

    /// <inheritdoc cref="AudioTemperature"/>
    public int AudioTopK { get; set; } = 250;

    /// <inheritdoc cref="AudioTemperature"/>
    public double TextTemperature { get; set; } = 0.7;

    /// <inheritdoc cref="AudioTemperature"/>
    public int TextTopK { get; set; } = 25;

    /// <summary>The TTS sampling temperature (§5.7: 0.6); ASR decodes greedily.</summary>
    public double TtsTemperature { get; set; } = 0.6;

    /// <summary>App. C's TTS pad-rate control: when the running fraction of padding steps falls below the target, the
    /// padding logits get this bonus.</summary>
    public double TtsPaddingTarget { get; set; } = 0.65;

    /// <inheritdoc cref="TtsPaddingTarget"/>
    public double TtsPaddingBonus { get; set; } = 1.0;

    // ---------------------------------------------------------------- training

    /// <summary>The stage a language-model step follows.</summary>
    public MoshiTrainingStage Stage { get; set; } = MoshiTrainingStage.PreTraining;

    /// <summary>Pre-training's Depth Transformer rate (Table 1: 2e-4); <c>LearningRate</c> is its Temporal Transformer's
    /// (3e-5). The later stages use Table 1's: post-training 3e-6 and 5e-5, Fisher 2e-6 and 4e-6, instruct 2e-6 for
    /// both.</summary>
    public double DepthLearningRate { get; set; } = 2e-4;

    /// <summary>Steps of the pre-training cosine schedule (1M) and its linear warmup (not stated; 1000). The later stages
    /// hold their rates fixed.</summary>
    public int TotalSteps { get; set; } = 1_000_000;

    /// <inheritdoc cref="TotalSteps"/>
    public int WarmupSteps { get; set; } = 1000;

    /// <summary>Pre-training's text masking probability (0.3) and text delay range (±0.6 s).</summary>
    public double TextMaskProbability { get; set; } = 0.3;

    /// <inheritdoc cref="TextMaskProbability"/>
    public double TextDelaySeconds { get; set; } = 0.6;

    /// <summary>The text embedding's and text linear's rate multiplier on audio batches (0.75).</summary>
    public double TextRateMultiplier { get; set; } = 0.75;

    /// <summary>The weight of PAD and EPAD in the text loss (0.5) and of each speaker's semantic codebook in the audio loss
    /// (α = 100, Eq. 7).</summary>
    public double TextPaddingWeight { get; set; } = 0.5;

    /// <inheritdoc cref="TextPaddingWeight"/>
    public double SemanticWeight { get; set; } = 100.0;

    /// <summary>Gradient clipping (1.0).</summary>
    public double MaxGradientNorm { get; set; } = 1.0;
}
