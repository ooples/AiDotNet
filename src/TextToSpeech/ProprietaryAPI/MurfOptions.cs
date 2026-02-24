using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>Options for Murf TTS API wrapper.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Murf model. Default values follow the original paper settings.</para>
/// </remarks>
public class MurfOptions : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MurfOptions(MurfOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ApiKey = other.ApiKey;
        ApiEndpoint = other.ApiEndpoint;
        VoiceId = other.VoiceId;
    }

    public MurfOptions()
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
