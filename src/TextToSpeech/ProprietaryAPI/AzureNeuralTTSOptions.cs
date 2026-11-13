namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>Options for the Azure AI Speech text-to-speech client (<c>POST /cognitiveservices/v1</c>).</summary>
/// <remarks>
/// <para>Defaults: region eastus (endpoint https://eastus.tts.speech.microsoft.com), the voice en-US-JennyNeural, and
/// 24 kHz raw 16-bit mono PCM (<c>raw-24khz-16bit-mono-pcm</c>; 8, 16, 22.05, 24, 44.1 and 48 kHz are offered). The
/// resource key goes in <c>Ocp-Apim-Subscription-Key</c>; a bearer token can be used instead. A response is at most ten
/// minutes of audio.</para>
/// <para><b>For Beginners:</b> Set <see cref="CloudTtsOptions.ApiKey"/> to your Speech resource key and
/// <see cref="Region"/> to its region.</para>
/// </remarks>
public class AzureNeuralTTSOptions : CloudTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public AzureNeuralTTSOptions(AzureNeuralTTSOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        Region = other.Region;
        BearerToken = other.BearerToken;
        TextIsSsml = other.TextIsSsml;
        UserAgent = other.UserAgent;
    }

    /// <summary>Creates the default options.</summary>
    public AzureNeuralTTSOptions()
    {
        VoiceId = "en-US-JennyNeural";
        SampleRate = 24000;
    }

    /// <summary>Gets or sets the resource's region (eastus).</summary>
    public string Region { get; set; } = "eastus";

    /// <summary>Gets or sets a bearer token used instead of the resource key, or null.</summary>
    public string? BearerToken { get; set; }

    /// <summary>Gets or sets whether the text is a complete SSML document rather than plain text.</summary>
    public bool TextIsSsml { get; set; }

    /// <summary>Gets or sets the application name sent as User-Agent (required by the service).</summary>
    public string UserAgent { get; set; } = "AiDotNet";
}
