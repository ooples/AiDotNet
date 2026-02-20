using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>Options for StyleTTS2 TTS model.</summary>
public class StyleTTS2Options : EndToEndTtsOptions
{
    public StyleTTS2Options()
    {
        NumEncoderLayers = 6;
        NumDecoderLayers = 4;
        NumHeads = 2;
        DropoutRate = 0.1;
    }

    public int StyleDim { get; set; } = 128;
    public int NumStyleDiffusionSteps { get; set; } = 5;
}
