namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for CosyVoice 2.</summary>
/// <remarks>
/// <para>The defaults are CosyVoice2-0.5B's (FunAudioLLM/CosyVoice <c>examples/libritts/cosyvoice2/conf/cosyvoice2.yaml</c>):
/// Qwen2.5-0.5B as the language model (24 layers, 896 wide, 14 query and 2 key-value heads, 4,864 units, RoPE θ 10⁶,
/// RMSNorm ε 10⁻⁶; a 151,936-row token table read through Qwen's tokenizer with CosyVoice 2's added tokens) speaking
/// the 25 Hz S3Tokenizer v2's 6,561 tokens; the causal flow (an upsampling Conformer, 512 wide, 6 + 4 blocks, chunks of
/// 25 tokens; a causal U-Net estimator, channels 256, 4 blocks, 12 mid blocks) giving two 80-band mel frames per token;
/// HiFT at 24 kHz (upsampling 8·5·3, iSTFT 16/4); a 192-wide CAM++ x-vector.</para>
/// <para><b>For Beginners:</b> These options configure the CosyVoice 2 model. Default values follow the released model.</para>
/// </remarks>
public class CosyVoice2Options : CosyVoiceFamilyOptions
{
    public CosyVoice2Options()
    {
        SampleRate = 24000;
        MelChannels = 80;
        HopSize = 480;
        FftSize = 1920;
        NumCodebooks = 1;
        CodebookSize = 6561;
        CodecFrameRate = 25;
        VocabSize = 151936;
        LLMDim = 896;
        NumLLMLayers = 24;
        NumHeads = 14;
        SpeakerEmbeddingDim = 192;
        DropoutRate = 0.1;
        DecoderChannels = new[] { 256 };
        HiftUpsampleRates = new[] { 8, 5, 3 };
        HiftUpsampleKernels = new[] { 16, 11, 7 };
        HiftSourceKernels = new[] { 7, 7, 11 };
        // The released recipe fine-tunes from Qwen at a constant rate: no warmup.
        WarmupSteps = 0;
        VocoderSegmentLength = 24480;
    }

    /// <summary>Initializes a new instance by copying all CosyVoice 2 settings.</summary>
    public CosyVoice2Options(CosyVoice2Options other)
        : base(other)
    {
        KeyValueHeads = other.KeyValueHeads;
        LmFeedForwardDim = other.LmFeedForwardDim;
        RopeTheta = other.RopeTheta;
        RmsEpsilon = other.RmsEpsilon;
        StaticChunkSize = other.StaticChunkSize;
        TokenMelRatio = other.TokenMelRatio;
        Lookahead = other.Lookahead;
    }

    /// <summary>Qwen2's key-value heads (grouped-query attention).</summary>
    public int KeyValueHeads { get; set; } = 2;

    /// <summary>Qwen2's feed-forward width (<c>intermediate_size</c>).</summary>
    public int LmFeedForwardDim { get; set; } = 4864;

    /// <summary>Qwen2's rotary base (<c>rope_theta</c>).</summary>
    public double RopeTheta { get; set; } = 1000000.0;

    /// <summary>Qwen2's RMSNorm epsilon.</summary>
    public double RmsEpsilon { get; set; } = 1e-6;

    /// <summary>The flow encoder's streaming chunk in tokens (<c>chunk_size</c>; the estimator's is this times
    /// <see cref="TokenMelRatio"/>).</summary>
    public int StaticChunkSize { get; set; } = 25;

    /// <summary>Mel frames per speech token (<c>token_mel_ratio</c>).</summary>
    public int TokenMelRatio { get; set; } = 2;

    /// <summary>Tokens the flow encoder's look-ahead convolution reads ahead (<c>pre_lookahead_len</c>).</summary>
    public int Lookahead { get; set; } = 3;
}
