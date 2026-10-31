namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>Options for the Murf text-to-speech client (<c>POST /v1/speech/generate</c>).</summary>
/// <remarks>
/// <para>Defaults: the endpoint https://api.murf.ai, the voice en-US-natalie, mono WAV at 24 kHz (Murf offers 8, 24, 44.1
/// and 48 kHz) returned as base64 (<c>encodeAsBase64</c>, so no file is kept on Murf's servers). The key goes in the
/// <c>api-key</c> header.</para>
/// <para><b>For Beginners:</b> Set <see cref="CloudTtsOptions.ApiKey"/> to your Murf API key.</para>
/// </remarks>
public class MurfOptions : CloudTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MurfOptions(MurfOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        Style = other.Style;
        Rate = other.Rate;
        Pitch = other.Pitch;
        Locale = other.Locale;
    }

    /// <summary>Creates the default options.</summary>
    public MurfOptions()
    {
        VoiceId = "en-US-natalie";
        SampleRate = 24000;
    }

    /// <summary>Gets or sets the voice style, or null for the voice's default.</summary>
    public string? Style { get; set; }

    /// <summary>Gets or sets the speed adjustment (−50–50), or null for 0.</summary>
    public int? Rate { get; set; }

    /// <summary>Gets or sets the pitch adjustment (−50–50), or null for 0.</summary>
    public int? Pitch { get; set; }

    /// <summary>Gets or sets the locale of a multilingual voice, or null.</summary>
    public string? Locale { get; set; }
}
