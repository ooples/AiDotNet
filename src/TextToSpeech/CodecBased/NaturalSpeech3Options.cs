using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for NaturalSpeech3 TTS model.</summary>
public class NaturalSpeech3Options : EndToEndTtsOptions
{
    public NaturalSpeech3Options()
    {
        NumDiffusionSteps = 100;
        NumEncoderLayers = 6;
        NumHeads = 4;
        DropoutRate = 0.1;
    }

    public int DiffusionDim { get; set; } = 256;
}
