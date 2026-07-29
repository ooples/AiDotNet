namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Options for XTTSv2Clone voice cloning model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the XTTSv2Clone model. Default values follow the original paper settings.</para>
/// </remarks>
public class XTTSv2CloneOptions : VoiceCloningOptions
{
    /// <summary>Initializes a copy of an XTTS v2 voice-cloning configuration.</summary>
    public XTTSv2CloneOptions(XTTSv2CloneOptions other)
        : base(other)
    {
        AdamBeta1 = other.AdamBeta1;
        AdamBeta2 = other.AdamBeta2;
        AdamEpsilon = other.AdamEpsilon;
    }

    public XTTSv2CloneOptions()
    {
        SampleRate = 24000;
        MinReferenceDurationSec = 6.0;
        NumCodebooks = 1;
        CodebookSize = 1024;
        CodecFrameRate = 22;
        TextEncoderDim = 1024;
        NumEncoderLayers = 0;
        LLMDim = 1024;
        NumLLMLayers = 30;
        NumHeads = 16;
        SpeakerEmbeddingDim = 512;
        LearningRate = 5e-6;
        WeightDecay = 1e-2;
        DropoutRate = 0.1;
        LanguageModelName = "GPT-2";
    }

    /// <summary>Gets or sets AdamW's first-moment coefficient.</summary>
    public double AdamBeta1 { get; set; } = 0.9;

    /// <summary>Gets or sets AdamW's second-moment coefficient.</summary>
    public double AdamBeta2 { get; set; } = 0.96;

    /// <summary>Gets or sets AdamW's numerical-stability epsilon.</summary>
    public double AdamEpsilon { get; set; } = 1e-8;
}
