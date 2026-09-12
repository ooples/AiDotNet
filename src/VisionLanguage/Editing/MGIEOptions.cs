namespace AiDotNet.VisionLanguage.Editing;

/// <summary>
/// Configuration options for MGIE: MLLM-guided image editing with LLaVA-based instruction understanding.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MGIE model. Default values follow the original paper settings.</para>
/// </remarks>
public class MGIEOptions : EditingVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MGIEOptions(MGIEOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        ImageSize = other.ImageSize;
        VisionDim = other.VisionDim;
        DecoderDim = other.DecoderDim;
        NumVisionLayers = other.NumVisionLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        EditHeadLayers = other.EditHeadLayers;
        NumHeads = other.NumHeads;
        VocabSize = other.VocabSize;
        MaxSequenceLength = other.MaxSequenceLength;
        MaxGenerationLength = other.MaxGenerationLength;
        DropoutRate = other.DropoutRate;
        ArchitectureType = other.ArchitectureType;
        ImageMean = other.ImageMean is null ? throw new ArgumentException("ImageMean is required.", nameof(other)) : (double[])other.ImageMean.Clone();
        ImageStd = other.ImageStd is null ? throw new ArgumentException("ImageStd is required.", nameof(other)) : (double[])other.ImageStd.Clone();
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        OutputImageSize = other.OutputImageSize;
        NumDiffusionSteps = other.NumDiffusionSteps;
        GuidanceScale = other.GuidanceScale;
        EnableExpressiveInstructions = other.EnableExpressiveInstructions;
        VisionPatchSize = other.VisionPatchSize;
        EditHiddenDim = other.EditHiddenDim;
        EditNumHeads = other.EditNumHeads;
        EditTokenCount = other.EditTokenCount;
        EditQueryCount = other.EditQueryCount;
        ImageGuidanceScale = other.ImageGuidanceScale;
    }

    public MGIEOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        EditHeadLayers = 4;
        NumHeads = 32;
        ImageSize = 512;
        VocabSize = 32000;
        MaxSequenceLength = 2048;
    }

    /// <summary>
    /// Gets or sets whether string editing requests first generate an expressive continuation
    /// with the native MLLM. Training/token-ID APIs consume the explicit caller-supplied tokens.
    /// Generation needs trained weights to produce meaningful language; construction does not load them.
    /// </summary>
    public bool EnableExpressiveInstructions { get; set; } = true;

    /// <summary>Vision encoder patch size; independent of the diffusion output resolution.</summary>
    public int VisionPatchSize { get; set; } = 14;

    /// <summary>Hidden width of the edit mapper, 512 in the released MGIE implementation.</summary>
    public int EditHiddenDim { get; set; } = 512;

    /// <summary>Attention heads in the mapper, independent of the language model's heads.</summary>
    public int EditNumHeads { get; set; } = 4;

    /// <summary>Learned edit tokens appended to the joint image/instruction stream.</summary>
    public int EditTokenCount { get; set; } = 8;

    /// <summary>Learned output queries, 77 in the released MGIE implementation.</summary>
    public int EditQueryCount { get; set; } = 77;

    /// <summary>Image-only classifier-free guidance scale; the text scale is GuidanceScale.</summary>
    public double ImageGuidanceScale { get; set; } = 1.5;
}
