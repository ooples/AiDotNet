namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>Options for the WellSaid Labs text-to-speech client (<c>POST /v1/tts/stream</c>).</summary>
/// <remarks>
/// <para>Defaults: the endpoint https://api.wellsaidlabs.com and speaker 3. The service returns MP3 only; the client decodes
/// it and resamples to <see cref="CloudTtsOptions.SampleRate"/> (24 kHz) when the stream's rate differs. The key goes in
/// the <c>X-Api-Key</c> header.</para>
/// <para><b>For Beginners:</b> Set <see cref="CloudTtsOptions.ApiKey"/> to your WellSaid key and
/// <see cref="CloudTtsOptions.VoiceId"/> to a speaker (voice avatar) id.</para>
/// </remarks>
public class WellSaidLabsOptions : CloudTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public WellSaidLabsOptions(WellSaidLabsOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        Model = other.Model;
    }

    /// <summary>Creates the default options.</summary>
    public WellSaidLabsOptions()
    {
        VoiceId = "3";
        SampleRate = 24000;
    }

    /// <summary>Gets or sets the model, or null for the service's default.</summary>
    public string? Model { get; set; }
}
