namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for PriorGrad (diffusion vocoder with data-dependent prior for adaptive noise).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the PriorGrad model. Default values follow the original paper settings.</para>
/// </remarks>
public class PriorGradOptions : VocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public PriorGradOptions(PriorGradOptions other)
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumResBlocks = other.NumResBlocks;
        OptimizerBatchSize = other.OptimizerBatchSize;
        OptimizerBeta1 = other.OptimizerBeta1;
        OptimizerBeta2 = other.OptimizerBeta2;
        OptimizerEpsilon = other.OptimizerEpsilon;
        MaxGradientNorm = other.MaxGradientNorm;
    }

    public PriorGradOptions()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        NumDiffusionSteps = 50;
        HiddenDim = 64;
        LearningRate = 2e-4;
        WeightDecay = 0.0;

        // PriorGrad's attention stack is 2 heads, and this has to be stated HERE rather than left
        // to the base. PriorGrad.InitializeLayers used to pass a hardcoded 2 to
        // CreateDefaultDiffusionVocoderLayers; replacing that with _options.NumHeads made the value
        // configurable, but nothing assigned it, so it silently inherited TtsModelOptions.NumHeads =
        // 8 and every default-constructed PriorGrad got a 4x wider attention stack than before.
        // That is a change to the trained architecture, not a refactor: existing checkpoints no
        // longer match the layer set they were trained into.
        NumHeads = 2;
    }

    /// <summary>Gets or sets the number of residual layers. The paper default is 30.</summary>
    public int NumResBlocks { get; set; } = 30;

    /// <summary>Gets or sets the Adam mini-batch size. The paper default is 16.</summary>
    public int OptimizerBatchSize { get; set; } = 16;

    /// <summary>Gets or sets Adam's first-moment decay.</summary>
    public double OptimizerBeta1 { get; set; } = 0.9;

    /// <summary>Gets or sets Adam's second-moment decay.</summary>
    public double OptimizerBeta2 { get; set; } = 0.999;

    /// <summary>Gets or sets Adam's numerical-stability epsilon.</summary>
    public double OptimizerEpsilon { get; set; } = 1e-8;

    /// <summary>
    /// Gets or sets the gradient clipping norm. A value less than or equal to zero disables
    /// clipping, matching the paper's default; users may set a positive value explicitly.
    /// </summary>
    public double MaxGradientNorm { get; set; } = 0.0;
}
