namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>Options for YourTTS (multilingual zero-shot multi-speaker VITS variant with speaker and language conditioning).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the YourTTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class YourTTSOptions : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public YourTTSOptions(YourTTSOptions other)
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SpeakerEmbeddingDim = other.SpeakerEmbeddingDim;
        NumLanguages = other.NumLanguages;
    }

    public YourTTSOptions()
    {
        SampleRate = 16000;
        MelChannels = 80;
        HopSize = 256;
        HiddenDim = 192;
        NumFlowSteps = 4;
        // VITS2 (Kong et al. 2023) and YourTTS (Casanova et al. 2022) both keep VITS's
        // optimizer recipe: AdamW, beta = (0.8, 0.99), weight decay 0.01, lr 2e-4.
        LearningRate = 2e-4;
        WeightDecay = 0.01;
    }

    public int SpeakerEmbeddingDim { get; set; } = 256;
    public int NumLanguages { get; set; } = 16;

    /// <summary>First AdamW moment coefficient used by the released YourTTS recipe.</summary>
    public double Beta1 { get; set; } = 0.8;

    /// <summary>Second AdamW moment coefficient used by the released YourTTS recipe.</summary>
    public double Beta2 { get; set; } = 0.99;

    /// <summary>AdamW numerical-stability epsilon used by the released YourTTS recipe.</summary>
    public double Epsilon { get; set; } = 1e-9;
}
