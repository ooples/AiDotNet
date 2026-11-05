namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for the Fun-CosyVoice3-0.5B-2512 release.</summary>
/// <remarks>The defaults reproduce the released Qwen2 LM, causal DiT flow and 24 kHz CausalHiFT.</remarks>
public class CosyVoice3Options : CosyVoice2Options
{
    /// <summary>Initializes a new instance with default values.</summary>
    public CosyVoice3Options()
    {
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
    public int DiTDim { get; set; }
    /// <summary>Gets or sets the diffusion transformer depth.</summary>
    public int DiTDepth { get; set; }
    /// <summary>Gets or sets the number of attention heads in the diffusion transformer.</summary>
    public int DiTHeads { get; set; }
    /// <summary>Gets or sets the feature width of the diffusion transformer head.</summary>
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
