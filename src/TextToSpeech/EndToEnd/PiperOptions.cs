namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>Options for Piper (lightweight VITS-based TTS optimized for edge/embedded deployment).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Piper model. Default values follow the original paper settings.</para>
/// </remarks>
public class PiperOptions : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public PiperOptions(PiperOptions other)
        : base(other)
    {
        LengthScale = other.LengthScale;
        NoiseScale = other.NoiseScale;
        NoiseScaleW = other.NoiseScaleW;
        OptimizerBeta1 = other.OptimizerBeta1;
        OptimizerBeta2 = other.OptimizerBeta2;
        OptimizerEpsilon = other.OptimizerEpsilon;
    }

    /// <summary>Initializes a new instance with default values.</summary>
    public PiperOptions()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        HiddenDim = 192;
        NumFlowSteps = 4;
        NumEncoderLayers = 4;
        LearningRate = 2e-4;
        WeightDecay = 0.01;
    }

    /// <summary>Gets or sets the multiplier on predicted phoneme durations; values above 1 speak more slowly.</summary>
    public double LengthScale { get; set; } = 1.0;
    /// <summary>Gets or sets the prior noise scale of the reference VITS voices. The native synthesis path does not read it.</summary>
    public double NoiseScale { get; set; } = 0.667;
    /// <summary>Gets or sets the duration-predictor noise scale of the reference VITS voices. The native synthesis path does not read it.</summary>
    public double NoiseScaleW { get; set; } = 0.8;

    /// <summary>Gets or sets AdamW's first-moment decay from the VITS training recipe.</summary>
    public double OptimizerBeta1 { get; set; } = 0.8;

    /// <summary>Gets or sets AdamW's second-moment decay from the VITS training recipe.</summary>
    public double OptimizerBeta2 { get; set; } = 0.99;

    /// <summary>Gets or sets AdamW's numerical-stability epsilon.</summary>
    public double OptimizerEpsilon { get; set; } = 1e-9;
}
