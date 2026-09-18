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
        ConditionDropoutProbability = other.ConditionDropoutProbability;
        InstructionLossWeight = other.InstructionLossWeight;
        EditLossWeight = other.EditLossWeight;
        TrainableLanguageModelScope = other.TrainableLanguageModelScope;
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

    /// <summary>
    /// Probability of dropping a training example's conditioning, for classifier-free guidance.
    /// </summary>
    /// <remarks>
    /// <para>Following InstructPix2Pix, MGIE removes the source image, the instruction guidance, or both for 5% of
    /// the training data (Fu et al. 2024, Sec. 3.3), so the denoiser also learns the image-only and unconditional
    /// branches that guided sampling combines.</para>
    /// <para><b>For Beginners:</b> occasionally hiding the hints teaches the model what an edit looks like without
    /// them, which is what lets guidance strengthen the edit at generation time.</para>
    /// </remarks>
    public double ConditionDropoutProbability { get; set; } = 0.05;

    /// <summary>Weight of the instruction (language-model) loss in the total objective.</summary>
    /// <remarks>
    /// <para>MGIE optimizes L_all = L_ins + 0.5 L_edit (Fu et al. 2024, Eq. 5), so this defaults to 1.</para>
    /// <para><b>For Beginners:</b> how strongly the language model is taught to write the expressive instruction.</para>
    /// </remarks>
    public double InstructionLossWeight { get; set; } = 1.0;

    /// <summary>Weight of the edit (diffusion noise-prediction) loss in the total objective.</summary>
    /// <remarks>
    /// <para>MGIE optimizes L_all = L_ins + 0.5 L_edit (Fu et al. 2024, Eq. 5), so this defaults to 0.5.</para>
    /// <para><b>For Beginners:</b> how strongly the image-editing part is taught to reproduce the target edit.</para>
    /// </remarks>
    public double EditLossWeight { get; set; } = 0.5;

    /// <summary>Which parts of the instruction language model training updates.</summary>
    /// <remarks>
    /// <para>MGIE keeps the MLLM frozen except its word embeddings and LM head (Fu et al. 2024, Sec. 4), which is the
    /// default.</para>
    /// <para><b>For Beginners:</b> the large language model mostly stays as it is; only the pieces that read and
    /// write tokens learn during editing training.</para>
    /// </remarks>
    public AiDotNet.Enums.LanguageModelTrainableScope TrainableLanguageModelScope { get; set; } =
        AiDotNet.Enums.LanguageModelTrainableScope.WordEmbeddingsAndHead;
}
