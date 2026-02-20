using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Options for E3TTS TTS model.</summary>
public class E3TTSOptions : EndToEndTtsOptions
{
    public E3TTSOptions()
    {
        NumDiffusionSteps = 50;
        NumEncoderLayers = 6;
        NumHeads = 4;
        DropoutRate = 0.1;
    }

    public int DiffusionDim { get; set; } = 256;
}
