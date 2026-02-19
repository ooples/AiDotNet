using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>Options for AudioPaLM TTS model.</summary>
public class AudioPaLMOptions : EndToEndTtsOptions
{
    public AudioPaLMOptions()
    {
        NumEncoderLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
