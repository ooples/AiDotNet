namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>Options for the NVIDIA Riva / Speech NIM text-to-speech client (<c>POST /v1/audio/synthesize</c>).</summary>
/// <remarks>
/// <para>Riva runs on your own GPU server (a TTS NIM container serves HTTP on port 9000), so the endpoint defaults to
/// http://localhost:9000. Defaults: language en-US, the voice Magpie-Multilingual.EN-US.Aria and 22.05 kHz (the Magpie
/// models' rate; the service returns 16-bit mono WAV). An API key, when the deployment requires one, is sent as a bearer
/// token.</para>
/// <para><b>For Beginners:</b> Start the TTS NIM container, then point <see cref="CloudTtsOptions.ApiEndpoint"/> at it.</para>
/// </remarks>
public class NVIDIARivaTTSOptions : CloudTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public NVIDIARivaTTSOptions(NVIDIARivaTTSOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        LanguageCode = other.LanguageCode;
    }

    /// <summary>Creates the default options.</summary>
    public NVIDIARivaTTSOptions()
    {
        ApiEndpoint = "http://localhost:9000";
        VoiceId = "Magpie-Multilingual.EN-US.Aria";
        SampleRate = 22050;
    }

    /// <summary>Gets or sets the BCP-47 language code (en-US).</summary>
    public string LanguageCode { get; set; } = "en-US";
}
