namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for AdaSpeech (adaptive TTS with acoustic condition modeling).</summary>
public class AdaSpeechOptions : AcousticModelOptions
{
    public AdaSpeechOptions() { EncoderDim = 256; DecoderDim = 80; HiddenDim = 256; NumEncoderLayers = 4; NumDecoderLayers = 4; NumHeads = 2; }

    /// <summary>Gets or sets the acoustic condition embedding dimension.</summary>
    public int ConditionDim { get; set; } = 256;

    /// <summary>Gets or sets the number of conditional layer normalization layers.</summary>
    public int NumConditionLayers { get; set; } = 4;
}
