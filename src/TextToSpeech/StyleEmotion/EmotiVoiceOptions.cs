using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>Options for EmotiVoice TTS model.</summary>
public class EmotiVoiceOptions : EndToEndTtsOptions
{
    public EmotiVoiceOptions()
    {
        NumEncoderLayers = 6;
        NumDecoderLayers = 4;
        NumHeads = 2;
        DropoutRate = 0.1;
    }

    public int EmotionDim { get; set; } = 128;
    public int NumEmotionLayers { get; set; } = 3;
}
