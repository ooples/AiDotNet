using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>Options for NVIDIARivaTTS TTS model.</summary>
public class NVIDIARivaTTSOptions : EndToEndTtsOptions
{
    public NVIDIARivaTTSOptions()
    {
        NumFlowSteps = 0;
        EncoderDim = 256;
        DecoderDim = 256;
        NumEncoderLayers = 4;
        NumDecoderLayers = 4;
        NumHeads = 4;
        DropoutRate = 0.1;
    }
}
