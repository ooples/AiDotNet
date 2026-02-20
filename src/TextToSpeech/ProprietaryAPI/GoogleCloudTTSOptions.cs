using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>Options for GoogleCloudTTS TTS API wrapper.</summary>
public class GoogleCloudTTSOptions : EndToEndTtsOptions
{
    public GoogleCloudTTSOptions()
    {
        NumFlowSteps = 0;
        NumEncoderLayers = 2;
        NumDecoderLayers = 2;
        NumHeads = 4;
        DropoutRate = 0.0;
    }

    /// <summary>API key for the Google Cloud TTS service.</summary>
    public string ApiKey { get; set; } = string.Empty;

    /// <summary>API endpoint URL for the Google Cloud TTS service.</summary>
    public string ApiEndpoint { get; set; } = string.Empty;

    /// <summary>Voice identifier to use for synthesis.</summary>
    public string VoiceId { get; set; } = "default";
}
