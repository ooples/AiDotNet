namespace AiDotNet.TextToSpeech.EndToEnd;
/// <summary>Options for YourTTS (multilingual zero-shot multi-speaker VITS variant with speaker and language conditioning).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the YourTTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class YourTTSOptions : EndToEndTtsOptions {
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public YourTTSOptions(YourTTSOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SpeakerEmbeddingDim = other.SpeakerEmbeddingDim;
        NumLanguages = other.NumLanguages;
    }
 public YourTTSOptions() { SampleRate = 16000; MelChannels = 80; HopSize = 256; HiddenDim = 192; NumFlowSteps = 4; } public int SpeakerEmbeddingDim { get; set; } = 256; public int NumLanguages { get; set; } = 16; }
