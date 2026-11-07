namespace AiDotNet.TextToSpeech.Latest;

/// <summary>The language model OuteTTS 1.0 is built on.</summary>
public enum OuteTTSBackbone
{
    /// <summary>Llama-OuteTTS-1.0-1B: Llama 3.2 1B (CC-BY-NC-SA; its tokenizer is read from the model's
    /// <c>tokenizer.json</c>).</summary>
    Llama1B,

    /// <summary>OuteTTS-1.0-0.6B: Qwen3 0.6B (Apache-2.0; its tokenizer is built in).</summary>
    Qwen3_0_6B,
}

/// <summary>Options for <see cref="OuteTTS{T}"/>.</summary>
/// <remarks>
/// <para>OuteTTS has no paper: the defaults are release 1.0 (edwko/OuteTTS and OuteAI/Llama-OuteTTS-1.0-1B's
/// <c>config.json</c> and <c>generation_config.json</c>): Llama 3.2 1B (width 2048, 16 layers, 32 heads over 8 key-value
/// heads, SwiGLU 8192, Llama 3 rotary scaling, tied embeddings, 134,400 tokens) speaking DAC codes
/// (ibm-research/DAC.speech.v1.0 at 24 kHz, 1.5 kbps: two codebooks of 1024 at 75 frames per second), sampled at
/// temperature 0.4 with a repetition penalty of 1.1 over the last 64 tokens, top-k 40, top-p 0.9 and min-p 0.05, in chunks
/// of 10–30 words. Speaker profiles time their words with Whisper large-v3-turbo. <see cref="Qwen3"/> gives
/// OuteTTS-1.0-0.6B.</para>
/// <para><b>For Beginners:</b> These settings size OuteTTS's language model, its audio codec and the Whisper model
/// that times a speaker's words; the defaults reproduce the released 1B model.</para>
/// </remarks>
public class OuteTTSOptions : TtsModelOptions
{
    /// <summary>Creates the Llama-OuteTTS-1.0-1B configuration.</summary>
    public OuteTTSOptions()
    {
        SampleRate = 24000;
        HopSize = 320;
        HiddenDim = 2048;
        VocabSize = 134400;
        MaxTextLength = 8192;
        LearningRate = 2e-5;
        WeightDecay = 0.0;
        DropoutRate = 0.0;
    }

    /// <summary>Copies another instance.</summary>
    public OuteTTSOptions(OuteTTSOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        Backbone = other.Backbone;
        Layers = other.Layers;
        Heads = other.Heads;
        KeyValueHeads = other.KeyValueHeads;
        HeadDim = other.HeadDim;
        Intermediate = other.Intermediate;
        RmsEpsilon = other.RmsEpsilon;
        RopeTheta = other.RopeTheta;
        RopeScalingFactor = other.RopeScalingFactor;
        RopeLowFrequencyFactor = other.RopeLowFrequencyFactor;
        RopeHighFrequencyFactor = other.RopeHighFrequencyFactor;
        RopeOriginalMaxPositions = other.RopeOriginalMaxPositions;
        QueryKeyNorm = other.QueryKeyNorm;
        Temperature = other.Temperature;
        RepetitionPenalty = other.RepetitionPenalty;
        RepetitionRange = other.RepetitionRange;
        TopK = other.TopK;
        TopP = other.TopP;
        MinP = other.MinP;
        MaxLength = other.MaxLength;
        Chunked = other.Chunked;
        ChunkMinWords = other.ChunkMinWords;
        ChunkMaxWords = other.ChunkMaxWords;
        CodecEncoderDim = other.CodecEncoderDim;
        CodecEncoderRates = (int[])other.CodecEncoderRates.Clone();
        CodecLatentDim = other.CodecLatentDim;
        CodecDecoderDim = other.CodecDecoderDim;
        CodecDecoderRates = (int[])other.CodecDecoderRates.Clone();
        CodecCodebooks = other.CodecCodebooks;
        CodecCodebookSize = other.CodecCodebookSize;
        CodecCodebookDim = other.CodecCodebookDim;
        AlignerMels = other.AlignerMels;
        AlignerWidth = other.AlignerWidth;
        AlignerHeads = other.AlignerHeads;
        AlignerEncoderLayers = other.AlignerEncoderLayers;
        AlignerDecoderLayers = other.AlignerDecoderLayers;
        AlignerTemperatures = (double[])other.AlignerTemperatures.Clone();
        MaxSpeakerSeconds = other.MaxSpeakerSeconds;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        Epsilon = other.Epsilon;
        MaxGradNorm = other.MaxGradNorm;
        TrainingSteps = other.TrainingSteps;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>OuteTTS-1.0-0.6B: Qwen3 0.6B (width 1024, 28 layers, 16 heads of width 128 over 8 key-value heads, SwiGLU
    /// 3072, per-head query and key RMSNorm, rotary base 10⁶, 157,760 tokens).</summary>
    public static OuteTTSOptions Qwen3() => new()
    {
        Backbone = OuteTTSBackbone.Qwen3_0_6B, HiddenDim = 1024, VocabSize = 157760, Layers = 28, Heads = 16, KeyValueHeads = 8, HeadDim = 128,
        Intermediate = 3072, RmsEpsilon = 1e-6, RopeTheta = 1000000, RopeScalingFactor = 0, QueryKeyNorm = true,
    };

    /// <summary>The language model (and with it the tokenizer).</summary>
    public OuteTTSBackbone Backbone { get; set; } = OuteTTSBackbone.Llama1B;

    /// <summary>Decoder layers (16).</summary>
    public int Layers { get; set; } = 16;

    /// <summary>Attention heads (32).</summary>
    public int Heads { get; set; } = 32;

    /// <summary>Key-value heads (8).</summary>
    public int KeyValueHeads { get; set; } = 8;

    /// <summary>Head width (64; 0 for width / heads).</summary>
    public int HeadDim { get; set; } = 64;

    /// <summary>SwiGLU width (8192).</summary>
    public int Intermediate { get; set; } = 8192;

    /// <summary>RMSNorm ε (1e-5).</summary>
    public double RmsEpsilon { get; set; } = 1e-5;

    /// <summary>Rotary base (500000).</summary>
    public double RopeTheta { get; set; } = 500000;

    /// <summary>Llama 3's rotary scaling factor (32; 0 for none).</summary>
    public double RopeScalingFactor { get; set; } = 32;

    /// <summary>Llama 3's low-frequency factor (1).</summary>
    public double RopeLowFrequencyFactor { get; set; } = 1;

    /// <summary>Llama 3's high-frequency factor (4).</summary>
    public double RopeHighFrequencyFactor { get; set; } = 4;

    /// <summary>Llama 3's original context (8192).</summary>
    public int RopeOriginalMaxPositions { get; set; } = 8192;

    /// <summary>Qwen3's per-head query and key RMSNorm (off for Llama).</summary>
    public bool QueryKeyNorm { get; set; }

    /// <summary>Sampling temperature (0.4; 0 samples greedily).</summary>
    public double Temperature { get; set; } = 0.4;

    /// <summary>Repetition penalty (1.1).</summary>
    public double RepetitionPenalty { get; set; } = 1.1;

    /// <summary>Tokens the repetition penalty looks back over (64).</summary>
    public int RepetitionRange { get; set; } = 64;

    /// <summary>Top-k (40).</summary>
    public int TopK { get; set; } = 40;

    /// <summary>Top-p (0.9).</summary>
    public double TopP { get; set; } = 0.9;

    /// <summary>Min-p (0.05).</summary>
    public double MinP { get; set; } = 0.05;

    /// <summary>Prompt and generated tokens at most (8192).</summary>
    public int MaxLength { get; set; } = 8192;

    /// <summary>Whether long text is generated in chunks (the release's default) rather than in one pass.</summary>
    public bool Chunked { get; set; } = true;

    /// <summary>Fewest words in a chunk (10).</summary>
    public int ChunkMinWords { get; set; } = 10;

    /// <summary>Most words in a chunk (30).</summary>
    public int ChunkMaxWords { get; set; } = 30;

    /// <summary>The codec encoder's first width (64).</summary>
    public int CodecEncoderDim { get; set; } = 64;

    /// <summary>The codec encoder's strides (2, 4, 5, 8: a hop of 320, 75 frames per second at 24 kHz).</summary>
    public int[] CodecEncoderRates { get; set; } = { 2, 4, 5, 8 };

    /// <summary>The codec latent width (1024).</summary>
    public int CodecLatentDim { get; set; } = 1024;

    /// <summary>The codec decoder's first width (1536).</summary>
    public int CodecDecoderDim { get; set; } = 1536;

    /// <summary>The codec decoder's upsampling (8, 5, 4, 2).</summary>
    public int[] CodecDecoderRates { get; set; } = { 8, 5, 4, 2 };

    /// <summary>Codebooks the language model speaks (2).</summary>
    public int CodecCodebooks { get; set; } = 2;

    /// <summary>Entries per codebook (1024).</summary>
    public int CodecCodebookSize { get; set; } = 1024;

    /// <summary>Codebook width (8).</summary>
    public int CodecCodebookDim { get; set; } = 8;

    /// <summary>The Whisper aligner's mel bins (128, large-v3-turbo).</summary>
    public int AlignerMels { get; set; } = 128;

    /// <summary>The Whisper aligner's width (1280).</summary>
    public int AlignerWidth { get; set; } = 1280;

    /// <summary>The Whisper aligner's heads (20).</summary>
    public int AlignerHeads { get; set; } = 20;

    /// <summary>The Whisper aligner's encoder layers (32).</summary>
    public int AlignerEncoderLayers { get; set; } = 32;

    /// <summary>The Whisper aligner's decoder layers (4).</summary>
    public int AlignerDecoderLayers { get; set; } = 4;

    /// <summary>The temperatures Whisper falls back through when transcribing a speaker (0, 0.2, …, 1).</summary>
    public double[] AlignerTemperatures { get; set; } = { 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 };

    /// <summary>The longest speaker recording, in seconds (20).</summary>
    public double MaxSpeakerSeconds { get; set; } = 20;

    /// <summary>AdamW β₁ (0.9).</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>AdamW β₂ (0.999).</summary>
    public double Beta2 { get; set; } = 0.999;

    /// <summary>AdamW ε (1e-8).</summary>
    public double Epsilon { get; set; } = 1e-8;

    /// <summary>The gradient norm clipped to (1.0).</summary>
    public double MaxGradNorm { get; set; } = 1.0;

    /// <summary>Updates the learning rate decays linearly to 0 over (0: constant). Set it to your run's length.</summary>
    public int TrainingSteps { get; set; }

    /// <summary>Seed of sampling and of the speaker transcriptions' fallback draws.</summary>
    public int SamplingSeed { get; set; }
}
