namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for Grad-TTS (diffusion-based acoustic model with score matching).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the GradTTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class GradTTSOptions : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GradTTSOptions(GradTTSOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumDiffusionSteps = other.NumDiffusionSteps;
        BetaStart = other.BetaStart;
        BetaEnd = other.BetaEnd;
    }

    public GradTTSOptions() { EncoderDim = 192; DecoderDim = 80; HiddenDim = 192; NumEncoderLayers = 6; NumDecoderLayers = 4; NumHeads = 2; }

    /// <summary>Gets or sets the number of diffusion steps at inference.</summary>
    public int NumDiffusionSteps { get; set; } = 10;

    /// <summary>Gets or sets the noise schedule beta start.</summary>
    public double BetaStart { get; set; } = 0.05;

    /// <summary>Gets or sets the noise schedule beta end.</summary>
    public double BetaEnd { get; set; } = 20.0;
}
