using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Options for DiTToTTS TTS model.</summary>
public class DiTToTTSOptions : EndToEndTtsOptions
{
    public DiTToTTSOptions()
    {
        NumEncoderLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        DropoutRate = 0.1;
    }

    public int FlowDim { get; set; } = 256;
    public int NumFlowLayers { get; set; } = 4;
}
