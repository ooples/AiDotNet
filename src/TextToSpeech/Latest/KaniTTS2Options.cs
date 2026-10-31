namespace AiDotNet.TextToSpeech.Latest;

/// <summary>Options for <see cref="KaniTTS2{T}"/>.</summary>
/// <remarks>
/// <para>KaniTTS 2 has no paper: the defaults are its release (nineninesix-ai/kani-tts-2 and the kani-tts 1.0.1 package;
/// nineninesix/kani-tts-2-en's <c>config.json</c>; nineninesix-ai/kani-tts-2-pretrain for training). It is Kani 1's LFM2-350M
/// and NanoCodec with three changes: frame-level positions (a frame's four codes share one rotary position, frames one
/// apart), a learnable rotary scale per attention layer (α in [0.1, 2]), and a 128-wide speaker embedding — WavLM-Large
/// speaker embeddings (nineninesix/speaker-emb-tbr, after Orange/Speaker-wavLM-tbr) — projected and inserted after the
/// first token; its vocabulary is 80,538 and a prompt may start with a language tag. Sampling is the package's (temperature
/// 1, top-k 50, top-p 0.95, repetition penalty 1.1, up to 3000 tokens). Training is the pretraining recipe: every
/// language-model weight with AdamW at 9e-4 (the rotary scales at 1.5 times that, without weight decay), 2000 warm-up
/// steps then cosine decay, the gradient norm clipped to 3, and 36 samples per step (6 × 6 accumulation).</para>
/// <para><b>For Beginners:</b> These settings size KaniTTS 2's language model, its audio codec and the speaker embedder
/// that reads a voice from a recording; the defaults reproduce the released model.</para>
/// </remarks>
public class KaniTTS2Options : KaniTTSOptions
{
    /// <summary>Creates the released configuration.</summary>
    public KaniTTS2Options()
    {
        VocabSize = 80538;
        MaxNewTokens = 3000;
        FrameLevelPositions = true;
        LearnableRope = true;
        SpeakerEmbeddingDim = 128;
        UseLora = false;
        LearningRate = 9e-4;
        WeightDecay = 0.0;
        MaxGradNorm = 3.0;
        WarmupSteps = 2000;
        GradientAccumulationSteps = 36;
    }

    /// <summary>Copies another instance.</summary>
    public KaniTTS2Options(KaniTTS2Options other)
        : base(other)
    {
        LanguageTags = (string[])other.LanguageTags.Clone();
        EmbedderConvDims = (int[])other.EmbedderConvDims.Clone();
        EmbedderWidth = other.EmbedderWidth;
        EmbedderLayers = other.EmbedderLayers;
        EmbedderHeads = other.EmbedderHeads;
        EmbedderIntermediate = other.EmbedderIntermediate;
        EmbedderPositionalKernel = other.EmbedderPositionalKernel;
        EmbedderPositionalGroups = other.EmbedderPositionalGroups;
        EmbedderHeadWidth = other.EmbedderHeadWidth;
        MaxReferenceSeconds = other.MaxReferenceSeconds;
    }

    /// <summary>The checkpoint's language tags (<c>config.json</c>'s <c>language_settings.language_tags_list</c>, which
    /// <c>LoadReferenceWeights</c> reads), indexed by a voice's or a training sample's <c>LanguageId</c>.</summary>
    public string[] LanguageTags { get; set; } = Array.Empty<string>();

    /// <summary>The speaker embedder's feature-encoder widths (seven of 512).</summary>
    public int[] EmbedderConvDims { get; set; } = { 512, 512, 512, 512, 512, 512, 512 };

    /// <summary>The speaker embedder's WavLM width (1024).</summary>
    public int EmbedderWidth { get; set; } = 1024;

    /// <summary>The speaker embedder's WavLM layers (24).</summary>
    public int EmbedderLayers { get; set; } = 24;

    /// <summary>The speaker embedder's WavLM heads (16).</summary>
    public int EmbedderHeads { get; set; } = 16;

    /// <summary>The speaker embedder's WavLM feed-forward width (4096).</summary>
    public int EmbedderIntermediate { get; set; } = 4096;

    /// <summary>The speaker embedder's position convolution kernel (128).</summary>
    public int EmbedderPositionalKernel { get; set; } = 128;

    /// <summary>The speaker embedder's position convolution groups (16).</summary>
    public int EmbedderPositionalGroups { get; set; } = 16;

    /// <summary>The speaker embedder head's middle width (512).</summary>
    public int EmbedderHeadWidth { get; set; } = 512;

    /// <summary>The longest speaker reference read, in seconds (30); longer recordings are cut.</summary>
    public double MaxReferenceSeconds { get; set; } = 30;
}
