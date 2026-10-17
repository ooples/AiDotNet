namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>Options for the ElevenLabs text-to-speech client (<c>POST /v1/text-to-speech/{voice_id}</c>).</summary>
/// <remarks>
/// <para>Defaults: the public endpoint https://api.elevenlabs.io, the <c>eleven_multilingual_v2</c> model, the premade
/// voice "Rachel" (21m00Tcm4TlvDq8ikWAM) and 24 kHz PCM (<c>output_format=pcm_24000</c>; the API's PCM rates are 8, 16,
/// 22.05, 24, 32, 44.1 and 48 kHz). The voice settings are left to the voice's defaults unless set.</para>
/// <para><b>For Beginners:</b> Set <see cref="CloudTtsOptions.ApiKey"/> to your ElevenLabs key and, optionally,
/// <see cref="CloudTtsOptions.VoiceId"/> to one of your voices.</para>
/// </remarks>
public class ElevenLabsTTSOptions : CloudTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ElevenLabsTTSOptions(ElevenLabsTTSOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        ModelId = other.ModelId;
        Stability = other.Stability;
        SimilarityBoost = other.SimilarityBoost;
        Style = other.Style;
        Speed = other.Speed;
        UseSpeakerBoost = other.UseSpeakerBoost;
    }

    /// <summary>Creates the default options.</summary>
    public ElevenLabsTTSOptions()
    {
        VoiceId = "21m00Tcm4TlvDq8ikWAM";
        SampleRate = 24000;
        MaxTextLength = 10000;
    }

    /// <summary>Gets or sets the model (<c>eleven_multilingual_v2</c>).</summary>
    public string ModelId { get; set; } = "eleven_multilingual_v2";

    /// <summary>Gets or sets the voice stability (0–1), or null for the voice's default.</summary>
    public double? Stability { get; set; }

    /// <summary>Gets or sets the similarity boost (0–1), or null for the voice's default.</summary>
    public double? SimilarityBoost { get; set; }

    /// <summary>Gets or sets the style exaggeration (0–1), or null for the voice's default.</summary>
    public double? Style { get; set; }

    /// <summary>Gets or sets the speaking speed, or null for the voice's default.</summary>
    public double? Speed { get; set; }

    /// <summary>Gets or sets whether to boost similarity to the original speaker, or null for the voice's default.</summary>
    public bool? UseSpeakerBoost { get; set; }
}
