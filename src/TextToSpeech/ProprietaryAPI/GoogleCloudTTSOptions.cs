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

    public string ApiKey { get; set; } = string.Empty;
    public string ApiEndpoint { get; set; } = string.Empty;
    public string VoiceId { get; set; } = "default";
}
