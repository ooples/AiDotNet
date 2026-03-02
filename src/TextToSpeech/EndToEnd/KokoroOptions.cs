namespace AiDotNet.TextToSpeech.EndToEnd;
/// <summary>Options for Kokoro (lightweight StyleTTS2-inspired TTS with style tokens and ISTFTNet decoder).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Kokoro model. Default values follow the original paper settings.</para>
/// </remarks>
public class KokoroOptions : EndToEndTtsOptions {
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public KokoroOptions(KokoroOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        StyleDim = other.StyleDim;
    }
 public KokoroOptions() { SampleRate = 24000; MelChannels = 80; HopSize = 256; HiddenDim = 512; NumFlowSteps = 0; } public int StyleDim { get; set; } = 128; }
