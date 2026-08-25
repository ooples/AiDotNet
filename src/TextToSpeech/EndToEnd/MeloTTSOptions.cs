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

    /// <summary>Validates the AdamW coefficients used by native MeloTTS training.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when a coefficient is non-finite or outside its valid range.</exception>
    internal void ValidateTrainingOptions()
    {
        if (!double.IsFinite(Beta1) || Beta1 < 0 || Beta1 >= 1)
            throw new ArgumentOutOfRangeException(nameof(Beta1), "Beta1 must be finite and in [0, 1).");
        if (!double.IsFinite(Beta2) || Beta2 < 0 || Beta2 >= 1)
            throw new ArgumentOutOfRangeException(nameof(Beta2), "Beta2 must be finite and in [0, 1).");
        if (!double.IsFinite(Epsilon) || Epsilon <= 0)
            throw new ArgumentOutOfRangeException(nameof(Epsilon), "Epsilon must be finite and positive.");
    }
}
