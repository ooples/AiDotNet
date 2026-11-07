namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for the Fun-CosyVoice3-0.5B-2512 release.</summary>
/// <remarks>The defaults reproduce the released Qwen2 LM, causal DiT flow and 24 kHz CausalHiFT.</remarks>
public class CosyVoice3Options : CosyVoiceFamilyOptions
{
    /// <summary>The released DiT flow produces exactly two mel frames per speech token.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("The released CosyVoice3 flow fixes TokenMelRatio at two. CosyVoice3OptionContextTests verifies three-token native training/inference produces six mel frames and rejects other ratios.")]
    public int TokenMelRatio { get; set; } = 2;

    /// <summary>Qwen2 key-value attention heads.</summary>
    public int KeyValueHeads { get; set; } = 2;
    /// <summary>Qwen2 feed-forward width.</summary>
    public int LmFeedForwardDim { get; set; } = 4864;
    /// <summary>Qwen2 rotary base.</summary>
    public double RopeTheta { get; set; } = 1000000.0;
    /// <summary>Qwen2 RMS normalization epsilon.</summary>
    public double RmsEpsilon { get; set; } = 1e-6;
    /// <summary>Streaming DiT chunk size in speech tokens.</summary>
    public int StaticChunkSize { get; set; } = 25;
    /// <summary>Speech tokens read ahead by the flow's pre-lookahead convolution.</summary>
    public int Lookahead { get; set; } = 3;
    /// <summary>Repetition-aware LM sampler nucleus probability.</summary>
    public double TopP { get; set; } = 0.8;
    /// <summary>Repetition-aware LM sampler candidate count.</summary>
    public int TopK { get; set; } = 25;
    /// <summary>Repetition-aware LM sampler history window.</summary>
    public int RepetitionWindow { get; set; } = 10;
    /// <summary>Repetition-aware LM sampler threshold ratio.</summary>
    public double RepetitionRatio { get; set; } = 0.1;

    /// <summary>Initializes a new instance with default values.</summary>
    public CosyVoice3Options()
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
        HiftUpsampleRates = new[] { 8, 5, 3 };
        HiftUpsampleKernels = new[] { 16, 11, 7 };
        HiftSourceKernels = new[] { 7, 7, 11 };
        WarmupSteps = 0;
        VocoderSegmentLength = 24480;
        DiTInputDim = 80;
        PreLookaheadChannels = 1024;
        DiTDim = 1024;
        DiTDepth = 22;
        DiTHeads = 16;
        DiTHeadDim = 64;
        DiTFeedForwardMultiplier = 2;
        DiTTimesteps = 10;
        DiTSigmaMin = 1e-6;
        DiTTrainingCfgRate = 0.2;
        DiTInferenceCfgRate = 0.7;
        ReservedSpeechTokens = 200;
        MaxReferenceSeconds = 30;
    }

    /// <summary>Initializes a copy.</summary>
    public CosyVoice3Options(CosyVoice3Options other)
        : base(other)
    {
        KeyValueHeads = other.KeyValueHeads;
        LmFeedForwardDim = other.LmFeedForwardDim;
        RopeTheta = other.RopeTheta;
        RmsEpsilon = other.RmsEpsilon;
        StaticChunkSize = other.StaticChunkSize;
        TokenMelRatio = other.TokenMelRatio;
        Lookahead = other.Lookahead;
        TopP = other.TopP;
        TopK = other.TopK;
        RepetitionWindow = other.RepetitionWindow;
        RepetitionRatio = other.RepetitionRatio;
        DiTInputDim = other.DiTInputDim;
        PreLookaheadChannels = other.PreLookaheadChannels;
        DiTDim = other.DiTDim;
        DiTDepth = other.DiTDepth;
        DiTHeads = other.DiTHeads;
        DiTHeadDim = other.DiTHeadDim;
        DiTFeedForwardMultiplier = other.DiTFeedForwardMultiplier;
        DiTTimesteps = other.DiTTimesteps;
        DiTSigmaMin = other.DiTSigmaMin;
        DiTTrainingCfgRate = other.DiTTrainingCfgRate;
        DiTInferenceCfgRate = other.DiTInferenceCfgRate;
        ReservedSpeechTokens = other.ReservedSpeechTokens;
        MaxReferenceSeconds = other.MaxReferenceSeconds;
        SpeechTokenizerPath = other.SpeechTokenizerPath;
        SpeakerEncoderOnnxPath = other.SpeakerEncoderOnnxPath;
    }

    /// <summary>Gets or sets the feature width of the diffusion transformer input.</summary>
    public int DiTInputDim { get; set; }
    /// <summary>Gets or sets the pre lookahead channels.</summary>
    public int PreLookaheadChannels { get; set; }
    /// <summary>Gets or sets the feature width of the diffusion transformer.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("DiTDim must equal DiTHeads times DiTHeadDim. CosyVoice3OptionContextTests runs native DiT training and inference at three valid paired geometries and rejects isolated changes.")]
    public int DiTDim { get; set; }
    /// <summary>Gets or sets the diffusion transformer depth.</summary>
    public int DiTDepth { get; set; }
    /// <summary>Gets or sets the number of attention heads in the diffusion transformer.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("DiTDim must equal DiTHeads times DiTHeadDim. CosyVoice3OptionContextTests runs native DiT training and inference at three valid paired geometries and rejects isolated changes.")]
    public int DiTHeads { get; set; }
    /// <summary>Gets or sets the feature width of the diffusion transformer head.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("DiTDim must equal DiTHeads times DiTHeadDim. CosyVoice3OptionContextTests runs native DiT training and inference at three valid paired geometries and rejects isolated changes.")]
    public int DiTHeadDim { get; set; }
    /// <summary>Gets or sets the diffusion transformer feed forward multiplier.</summary>
    public int DiTFeedForwardMultiplier { get; set; }
    /// <summary>Gets or sets the diffusion transformer timesteps.</summary>
    public int DiTTimesteps { get; set; }
    /// <summary>Gets or sets the diffusion transformer sigma min.</summary>
    public double DiTSigmaMin { get; set; }
    /// <summary>Gets or sets the diffusion transformer training classifier-free guidance rate.</summary>
    public double DiTTrainingCfgRate { get; set; }
    /// <summary>Gets or sets the diffusion transformer inference classifier-free guidance rate.</summary>
    public double DiTInferenceCfgRate { get; set; }

    /// <summary>Reserved speech ids after the S3 vocabulary; every one is a stop id during generation.</summary>
    public int ReservedSpeechTokens { get; set; }

}
