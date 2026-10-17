using AiDotNet.Audio.Generation;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for VALL-E (Wang et al. 2023, "Neural Codec Language Models are Zero-Shot Text to Speech
/// Synthesizers").</summary>
/// <remarks>
/// <para>
/// The defaults are the paper's (§5.1): EnCodec at 6 kbps (eight codebooks of 1024 codes, 75 frames a second, 24 kHz),
/// an AR and a NAR Transformer each of 12 layers, 16 heads, width 1024, feed-forward 4096 and dropout 0.1, trained with
/// AdamW warmed up over 32,000 updates to 5e-4 and decayed linearly over 800,000 updates; the NAR's training prompt is
/// a random 3-second segment of the same utterance. The paper converts text with a Kaldi ASR model's phoneme
/// alignments; this model reads espeak-style phonemes (<see cref="FrontEnd.EnglishG2P"/>), as the reference
/// reproduction (lifeiteng/vall-e) does, and takes the rest of what the paper leaves open from it: the phoneme table
/// (512 embedding rows), sinusoidal positions with a learned scale in the AR model, pre-norm layers, the NAR heads
/// sharing the acoustic embeddings, and sampling with temperature 1 and no top-k cut.
/// </para>
/// <para><b>For Beginners:</b> These options set the sizes of VALL-E's two networks, its codec and how it is trained
/// and sampled. The defaults reproduce the paper.</para>
/// </remarks>
public class VALLEOptions : TtsModelOptions
{
    /// <summary>Creates the paper's options.</summary>
    public VALLEOptions()
    {
        SampleRate = 24000;
        HopSize = 320;
        HiddenDim = 1024;
        NumHeads = 16;
        NumEncoderLayers = 12;
        NumDecoderLayers = 12;
        DropoutRate = 0.1;
        LearningRate = 5e-4;
        WeightDecay = 0.01;
        MaxTextLength = 512;
        Codec = CreatePaperCodec();
    }

    /// <summary>Creates a copy of <paramref name="other"/>.</summary>
    public VALLEOptions(VALLEOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        FeedForwardDim = other.FeedForwardDim;
        TextTokens = other.TextTokens;
        NumCodebooks = other.NumCodebooks;
        CodebookSize = other.CodebookSize;
        PromptSeconds = other.PromptSeconds;
        WarmupSteps = other.WarmupSteps;
        TrainingSteps = other.TrainingSteps;
        Temperature = other.Temperature;
        TopK = other.TopK;
        MaxCodesPerTextToken = other.MaxCodesPerTextToken;
        Codec = (EnCodecOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(other.Codec);
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Feed-forward width of both Transformers (4096). Their width is <see cref="TtsModelOptions.HiddenDim"/>,
    /// their heads <see cref="TtsModelOptions.NumHeads"/>, their layers <see cref="TtsModelOptions.NumDecoderLayers"/>
    /// (AR) and <see cref="TtsModelOptions.NumEncoderLayers"/> (NAR).</summary>
    public int FeedForwardDim { get; set; } = 4096;

    /// <summary>Rows of the phoneme embeddings (512, the reference's <c>NUM_TEXT_TOKENS</c>).</summary>
    public int TextTokens { get; set; } = 512;

    /// <summary>Codebooks (8: EnCodec at 6 kbps).</summary>
    public int NumCodebooks { get; set; } = 8;

    /// <summary>Codes per codebook (1024).</summary>
    public int CodebookSize { get; set; } = 1024;

    /// <summary>Length of the NAR's training prompt, in seconds (3, §5.1).</summary>
    public double PromptSeconds { get; set; } = 3.0;

    /// <summary>Linear warm-up of the learning rate, in updates (32,000, §5.1).</summary>
    public int WarmupSteps { get; set; } = 32_000;

    /// <summary>Updates over which the learning rate decays linearly to zero (800,000, §5.1).</summary>
    public int TrainingSteps { get; set; } = 800_000;

    /// <summary>AR sampling temperature (1, the reference's default; the paper says only "sampling-based").</summary>
    public double Temperature { get; set; } = 1.0;

    /// <summary>AR top-k cut (0 keeps every code, the reference's default).</summary>
    public int TopK { get; set; }

    /// <summary>The AR model stops after this many codes per phoneme token (16, the reference's limit).</summary>
    public int MaxCodesPerTextToken { get; set; } = 16;

    /// <summary>The codec (EnCodec 24 kHz at 6 kbps, the released checkpoint's configuration).</summary>
    public EnCodecOptions Codec { get; set; }

    /// <summary>Seed of the training draws (NAR stage, prompt segment) and of sampling.</summary>
    public int SamplingSeed { get; set; }

    /// <summary>EnCodec's released 24 kHz model at 6 kbps: eight codebooks.</summary>
    public static EnCodecOptions CreatePaperCodec()
    {
        var codec = EnCodecOptions.OfficialCheckpoint24kHz();
        codec.TargetBandwidthKbps = 6.0;
        return codec;
    }
}
