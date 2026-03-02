namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for ProDiff (progressive fast diffusion model for high-quality TTS).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the ProDiff model. Default values follow the original paper settings.</para>
/// </remarks>
public class ProDiffOptions : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ProDiffOptions(ProDiffOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumDiffusionSteps = other.NumDiffusionSteps;
        UseProgressiveDistillation = other.UseProgressiveDistillation;
    }

    public ProDiffOptions() { EncoderDim = 256; DecoderDim = 80; HiddenDim = 256; NumEncoderLayers = 4; NumDecoderLayers = 4; NumHeads = 2; }

    /// <summary>Gets or sets the number of diffusion steps at inference (progressive reduces to 2-4).</summary>
    public int NumDiffusionSteps { get; set; } = 4;

    /// <summary>Gets or sets whether to use knowledge distillation for step reduction.</summary>
    public bool UseProgressiveDistillation { get; set; } = true;
}
