namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>Options for the Google Cloud Text-to-Speech client (<c>POST /v1/text:synthesize</c>).</summary>
/// <remarks>
/// <para>Defaults: the endpoint https://texttospeech.googleapis.com, the voice en-US-Neural2-F (its language code is the
/// name's first two parts), and LINEAR16 at 24 kHz, which the service returns as a WAV file (base64 in the JSON
/// <c>audioContent</c>). Authentication is an API key (<c>X-Goog-Api-Key</c>) or an OAuth access token (bearer). The
/// service accepts at most 5,000 bytes of input.</para>
/// <para><b>For Beginners:</b> Set <see cref="CloudTtsOptions.ApiKey"/> to an API key restricted to the Text-to-Speech
/// API, or <see cref="AccessToken"/> to an OAuth token.</para>
/// </remarks>
public class GoogleCloudTTSOptions : CloudTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GoogleCloudTTSOptions(GoogleCloudTTSOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        AccessToken = other.AccessToken;
        LanguageCode = other.LanguageCode;
        SpeakingRate = other.SpeakingRate;
        Pitch = other.Pitch;
        TextIsSsml = other.TextIsSsml;
    }

    /// <summary>Creates the default options.</summary>
    public GoogleCloudTTSOptions()
    {
        VoiceId = "en-US-Neural2-F";
        SampleRate = 24000;
        MaxTextLength = 5000;
    }

    /// <summary>Gets or sets an OAuth access token used instead of the API key, or null.</summary>
    public string? AccessToken { get; set; }

    /// <summary>Gets or sets the language code, or null to take it from the voice name.</summary>
    public string? LanguageCode { get; set; }

    /// <summary>Gets or sets the speaking rate (0.25–4), or null for 1.</summary>
    public double? SpeakingRate { get; set; }

    /// <summary>Gets or sets the pitch in semitones (−20–20), or null for 0.</summary>
    public double? Pitch { get; set; }

    /// <summary>Gets or sets whether the text is SSML rather than plain text.</summary>
    public bool TextIsSsml { get; set; }
}
