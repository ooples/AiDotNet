using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>Options for NVIDIARivaTTS TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the NVIDIARivaTTS model. Default values follow the original paper settings.</para>
/// </remarks>
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
