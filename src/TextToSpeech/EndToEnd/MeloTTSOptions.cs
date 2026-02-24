namespace AiDotNet.TextToSpeech.EndToEnd;
/// <summary>Options for MeloTTS (multilingual VITS-based TTS with BERT-enhanced text processing and mixed-language support).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MeloTTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class MeloTTSOptions : EndToEndTtsOptions {
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MeloTTSOptions(MeloTTSOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SpeedFactor = other.SpeedFactor;
    }
 public MeloTTSOptions() { SampleRate = 44100; MelChannels = 80; HopSize = 512; HiddenDim = 192; NumFlowSteps = 4; } public double SpeedFactor { get; set; } = 1.0; }
