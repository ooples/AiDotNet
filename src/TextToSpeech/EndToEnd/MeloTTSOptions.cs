namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>Options for MeloTTS (multilingual VITS-based TTS with BERT-enhanced text processing and mixed-language support).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MeloTTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class MeloTTSOptions : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MeloTTSOptions(MeloTTSOptions other)
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SpeedFactor = other.SpeedFactor;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        Epsilon = other.Epsilon;
    }

    public MeloTTSOptions()
    {
        SampleRate = 44100;
        MelChannels = 80;
        HopSize = 512;
        HiddenDim = 192;
        NumFlowSteps = 4;
        LearningRate = 3e-4;
        WeightDecay = 0.01;
    }

    public double SpeedFactor { get; set; } = 1.0;

    /// <summary>First AdamW moment coefficient used by the released MeloTTS recipe.</summary>
    public double Beta1 { get; set; } = 0.8;

    /// <summary>Second AdamW moment coefficient used by the released MeloTTS recipe.</summary>
    public double Beta2 { get; set; } = 0.99;

    /// <summary>AdamW numerical-stability epsilon used by the released MeloTTS recipe.</summary>
    public double Epsilon { get; set; } = 1e-9;
}
