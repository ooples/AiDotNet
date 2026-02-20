using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>Options for PlayHT TTS API wrapper.</summary>
public class PlayHTOptions : EndToEndTtsOptions
{
    public PlayHTOptions()
    {
        NumFlowSteps = 0;
        NumEncoderLayers = 2;
        NumDecoderLayers = 2;
        NumHeads = 4;
        DropoutRate = 0.0;
    }

    /// <summary>API key for the PlayHT TTS service.</summary>
    public string ApiKey { get; set; } = string.Empty;

    /// <summary>API endpoint URL for the PlayHT TTS service.</summary>
    public string ApiEndpoint { get; set; } = string.Empty;

    /// <summary>Voice identifier to use for synthesis.</summary>
    public string VoiceId { get; set; } = "default";
}
