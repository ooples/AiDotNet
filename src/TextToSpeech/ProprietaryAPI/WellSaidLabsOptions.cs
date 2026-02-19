using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>Options for WellSaidLabs TTS API wrapper.</summary>
public class WellSaidLabsOptions : EndToEndTtsOptions
{
    public WellSaidLabsOptions()
    {
        NumFlowSteps = 0;
        NumEncoderLayers = 2;
        NumDecoderLayers = 2;
        NumHeads = 4;
        DropoutRate = 0.0;
    }

    public string ApiKey { get; set; } = string.Empty;
    public string ApiEndpoint { get; set; } = string.Empty;
    public string VoiceId { get; set; } = "default";
}
