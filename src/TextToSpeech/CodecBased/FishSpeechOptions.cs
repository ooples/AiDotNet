namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for FishSpeech (Fish Audio, 2024) dual-AR architecture with GFSQ.</summary>
public class FishSpeechOptions : CodecTtsOptions
{
    public FishSpeechOptions() { SampleRate = 44100; NumCodebooks = 8; CodebookSize = 1024; CodecFrameRate = 42; LLMDim = 1024; NumLLMLayers = 24; LanguageModelName = "LLaMA"; }

    /// <summary>Gets or sets the number of GFSQ groups for grouped finite scalar quantization.</summary>
    public int NumGroups { get; set; } = 8;

    /// <summary>Gets or sets the sampling temperature for generation.</summary>
    public double Temperature { get; set; } = 0.7;

    /// <summary>Gets or sets the top-p (nucleus) sampling parameter.</summary>
    public double TopP { get; set; } = 0.8;

    /// <summary>Gets or sets the repetition penalty factor.</summary>
    public double RepetitionPenalty { get; set; } = 1.2;

    /// <summary>Gets or sets the minimum reference audio duration in seconds for voice cloning.</summary>
    public double MinReferenceSeconds { get; set; } = 3.0;
}
