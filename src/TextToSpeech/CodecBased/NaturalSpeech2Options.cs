using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for NaturalSpeech2 TTS model.</summary>
public class NaturalSpeech2Options : EndToEndTtsOptions
{
    public NaturalSpeech2Options()
    {
        NumDiffusionSteps = 100;
        NumEncoderLayers = 6;
        NumHeads = 4;
        DropoutRate = 0.1;
    }

    public int DiffusionDim { get; set; } = 256;
}
