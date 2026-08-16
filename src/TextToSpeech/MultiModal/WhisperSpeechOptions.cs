using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>Options for WhisperSpeech.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the WhisperSpeech model. Default values follow the original paper settings.</para>
/// </remarks>
public class WhisperSpeechOptions : CodecTtsOptions
{
    /// <summary>Initializes options with the WhisperSpeech reference-training defaults.</summary>
    public WhisperSpeechOptions()
    {
        SampleRate = 24000;
        NumCodebooks = 2;
        CodebookSize = 1024;
        CodecFrameRate = 50;
        LLMDim = 768;
        NumLLMLayers = 6;
        LearningRate = 1e-4;
        WeightDecay = 0.1;
    }

    /// <summary>Initializes a deep copy of another WhisperSpeech options instance.</summary>
    public WhisperSpeechOptions(WhisperSpeechOptions other)
        : base(other)
    {
        if (other is null)
            throw new ArgumentNullException(nameof(other));
        AdamBeta1 = other.AdamBeta1;
        AdamBeta2 = other.AdamBeta2;
        MaxGradientNorm = other.MaxGradientNorm;
    }

    /// <summary>Gets or sets AdamW's first-moment decay.</summary>
    public double AdamBeta1 { get; set; } = 0.9;

    /// <summary>Gets or sets AdamW's second-moment decay.</summary>
    public double AdamBeta2 { get; set; } = 0.95;

    /// <summary>Gets or sets the maximum global gradient norm used during training.</summary>
    public double MaxGradientNorm { get; set; } = 1.0;
}
