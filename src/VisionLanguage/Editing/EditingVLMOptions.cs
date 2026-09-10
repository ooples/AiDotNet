using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.VisionLanguage.Editing;

/// <summary>
/// Base configuration options for image editing VLMs.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Editing model. Default values follow the original paper settings.</para>
/// </remarks>
public class EditingVLMOptions : GenerativeVLMOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public EditingVLMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public EditingVLMOptions(EditingVLMOptions other)
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
        ImageMean = other.ImageMean;
        ImageStd = other.ImageStd;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        OutputImageSize = other.OutputImageSize;
        NumDiffusionSteps = other.NumDiffusionSteps;
        GuidanceScale = other.GuidanceScale;
    }

    /// <summary>Gets or sets the output image resolution.</summary>
    public int OutputImageSize { get; set; } = 512;

    /// <summary>Gets or sets the number of diffusion denoising steps.</summary>
    public int NumDiffusionSteps { get; set; } = 50;

    /// <summary>Gets or sets the guidance scale for classifier-free guidance.</summary>
    public double GuidanceScale { get; set; } = 7.5;

    /// <summary>
    /// Depth of the edit head that carries MLLM visual tokens into the diffusion model.
    /// Paper default 4: MGIE (arXiv 2309.17102, Sec. 3) states "The edit head T is a 4-layer
    /// Transformer, which transforms language features into editing guidance"; EmuEdit and
    /// SmartEdit use the same bridging stage. Settable, so a caller can depart from the paper.
    ///
    /// All three models previously passed NumDecoderLayers here - the LLM's own depth, 32 for
    /// the LLaMA-7B in LLaVA-7B - building an edit head eight times the paper's depth. The
    /// other defaults are already paper-faithful: VisionDim 1024 with 24 layers is CLIP
    /// ViT-L/14 and DecoderDim 4096 with 32 layers and 32 heads is LLaMA-7B, which together
    /// are exactly the LLaVA-7B the paper initializes from.
    /// </summary>
    public int EditHeadLayers { get; set; } = 4;
}
