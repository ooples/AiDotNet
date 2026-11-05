namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>Options for Kokoro (lightweight StyleTTS2-inspired TTS with style tokens and ISTFTNet decoder).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Kokoro model. Default values follow the original paper settings.</para>
/// </remarks>
public class KokoroOptions : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public KokoroOptions(KokoroOptions other)
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        StyleDim = other.StyleDim;
        Speed = other.Speed;
        SynthesisSeed = other.SynthesisSeed;
        UseNativeEnglishFallback = other.UseNativeEnglishFallback;
    }

    /// <summary>Initializes a new instance with default values.</summary>
    public KokoroOptions()
    {
        SampleRate = 24000;
        MelChannels = 80;
        HopSize = 300;
        HiddenDim = 512;
        NumFlowSteps = 0;
        VocabSize = 178;
        NumEncoderLayers = 3;
        DropoutRate = .2;
    }

    /// <summary>Gets or sets the width of the style vector that conditions synthesis.</summary>
    public int StyleDim { get; set; } = 128;

    /// <summary>Duration scaling used by the released model: sum(sigmoid(duration logits)) / speed.</summary>
    public double Speed { get; set; } = 1;

    /// <summary>Optional per-call seed for reproducible native excitation; null keeps the advancing random stream.</summary>
    public int? SynthesisSeed { get; set; }
    /// <summary>Explicitly opts into the repository's approximate English frontend. Faithful raw text processing uses the official Misaki frontend through TextPhonemizer.</summary>
    public bool UseNativeEnglishFallback { get; set; }
}
