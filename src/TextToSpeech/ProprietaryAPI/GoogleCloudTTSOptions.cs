using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>Options for GoogleCloudTTS TTS API wrapper.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the GoogleCloudTTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class GoogleCloudTTSOptions : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GoogleCloudTTSOptions(GoogleCloudTTSOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ApiKey = other.ApiKey;
        ApiEndpoint = other.ApiEndpoint;
        VoiceId = other.VoiceId;
    }

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
