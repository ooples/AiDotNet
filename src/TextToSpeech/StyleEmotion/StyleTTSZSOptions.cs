using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>Options for StyleTTSZS TTS model.</summary>
public class StyleTTSZSOptions : EndToEndTtsOptions
{
    public StyleTTSZSOptions()
    {
        NumEncoderLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        DropoutRate = 0.1;
    }

    public int StyleDim { get; set; } = 256;
    public int NumStyleLayers { get; set; } = 4;
}
