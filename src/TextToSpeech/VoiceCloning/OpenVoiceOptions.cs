using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Options for OpenVoice TTS model.</summary>
public class OpenVoiceOptions : EndToEndTtsOptions
{
    public OpenVoiceOptions()
    {
        NumEncoderLayers = 6;
        NumDecoderLayers = 4;
        NumHeads = 2;
        DropoutRate = 0.1;
    }

    public int SpeakerEmbeddingDim { get; set; } = 256;
    public int NumToneColorLayers { get; set; } = 3;
}
