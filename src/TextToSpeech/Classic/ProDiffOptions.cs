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
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumDiffusionSteps = other.NumDiffusionSteps;
        UseProgressiveDistillation = other.UseProgressiveDistillation;
        WarmupSteps = other.WarmupSteps;
        OptimizerBeta1 = other.OptimizerBeta1;
        OptimizerBeta2 = other.OptimizerBeta2;
        OptimizerEpsilon = other.OptimizerEpsilon;
        MaxGradientNorm = other.MaxGradientNorm;
    }

    public ProDiffOptions()
    {
        EncoderDim = 256;
        DecoderDim = 80;
        HiddenDim = 256;
        NumEncoderLayers = 4;
        NumDecoderLayers = 4;
        NumHeads = 2;
        // The official ProDiff recipe uses the inherited FastSpeech 2
        // inverse-square-root schedule with factor 1.0 and 2,000 warmup steps.
        // For ProDiff, LearningRate is therefore the schedule factor rather
        // than a raw constant step size.
        LearningRate = 1.0;
        WeightDecay = 0.0;
    }

    /// <summary>Gets or sets the number of diffusion steps at inference (progressive reduces to 2-4).</summary>
    public int NumDiffusionSteps { get; set; } = 4;

    /// <summary>Gets or sets whether to use knowledge distillation for step reduction.</summary>
    public bool UseProgressiveDistillation { get; set; } = true;

    /// <summary>Gets or sets the number of inverse-square-root scheduler warmup steps.</summary>
    public int WarmupSteps { get; set; } = 2000;

    /// <summary>Gets or sets Adam's first-moment decay coefficient.</summary>
    public double OptimizerBeta1 { get; set; } = 0.9;

    /// <summary>Gets or sets Adam's second-moment decay coefficient.</summary>
    public double OptimizerBeta2 { get; set; } = 0.98;

    /// <summary>Gets or sets Adam's numerical-stability epsilon.</summary>
    public double OptimizerEpsilon { get; set; } = 1e-8;

    /// <summary>
    /// Gets or sets the global gradient-norm limit. Set to zero to disable clipping.
    /// </summary>
    public double MaxGradientNorm { get; set; } = 1.0;
}
