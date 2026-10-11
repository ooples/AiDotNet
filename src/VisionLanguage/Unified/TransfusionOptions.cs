namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Configuration options for Transfusion: combined autoregressive and diffusion loss in single transformer.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Transfusion model. Default values follow the original paper settings.</para>
/// </remarks>
public class TransfusionOptions : UnifiedVisionOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public TransfusionOptions(TransfusionOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        ImageSize = other.ImageSize;
        VisionDim = other.VisionDim;
        DecoderDim = other.DecoderDim;
        NumVisionLayers = other.NumVisionLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        NumHeads = other.NumHeads;
        VocabSize = other.VocabSize;
        MaxSequenceLength = other.MaxSequenceLength;
        MaxGenerationLength = other.MaxGenerationLength;
        DropoutRate = other.DropoutRate;
        ArchitectureType = other.ArchitectureType;
        ImageMean = other.ImageMean;
        ImageStd = other.ImageStd;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        LanguageModelName = other.LanguageModelName;
        SupportsGeneration = other.SupportsGeneration;
        OutputImageSize = other.OutputImageSize;
        NumVisualTokens = other.NumVisualTokens;
        EnableDiffusionLoss = other.EnableDiffusionLoss;
        DiffusionLossWeight = other.DiffusionLossWeight;
        DiffusionTimesteps = other.DiffusionTimesteps;
        BeginningOfImageTokenId = other.BeginningOfImageTokenId;
        EndOfImageTokenId = other.EndOfImageTokenId;
    }

    public TransfusionOptions()
    {
        VisionDim = 4096;
        DecoderDim = 4096;
        NumVisionLayers = 0;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 256;
        VocabSize = 32000;
        LanguageModelName = "Transfusion";
        NumVisualTokens = 256;
    }

    /// <summary>Gets or sets whether to use diffusion loss for image generation.</summary>
    public bool EnableDiffusionLoss { get; set; } = true;

    /// <summary>
    /// Gets or sets <c>lambda</c>, the coefficient that balances the diffusion loss against the
    /// language-modelling loss in <c>L = L_LM + lambda * L_DDPM</c>.
    /// </summary>
    /// <value>Default 5.0.</value>
    /// <remarks>Zhou et al. 2024, Sec. 4.1 (Optimization): "We set the lambda coefficient in the
    /// Transfusion objective (Equation 4) to 5 following preliminary experiments; we leave further
    /// tuning of lambda to future work."</remarks>
    public double DiffusionLossWeight { get; set; } = 5.0;

    /// <summary>Gets or sets the number of training timesteps of the diffusion process.</summary>
    /// <value>Default 1000.</value>
    /// <remarks>Zhou et al. 2024, Sec. 4.1 (Inference): "the model is trained on 1,000
    /// timesteps".</remarks>
    public int DiffusionTimesteps { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the beginning-of-image marker token id that precedes an image's patches.
    /// </summary>
    /// <value>Nullable. When null, the second-to-last id of the vocabulary is used.</value>
    /// <remarks>Zhou et al. 2024, Sec. 3 (Data Representation): each image sequence is surrounded
    /// by special BOI and EOI tokens before it is inserted into the text sequence.</remarks>
    public int? BeginningOfImageTokenId { get; set; }

    /// <summary>Gets or sets the end-of-image marker token id that follows an image's patches.</summary>
    /// <value>Nullable. When null, the last id of the vocabulary is used.</value>
    public int? EndOfImageTokenId { get; set; }
}
