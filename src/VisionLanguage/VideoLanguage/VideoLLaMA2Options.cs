namespace AiDotNet.VisionLanguage.VideoLanguage;

/// <summary>
/// Configuration options for VideoLLaMA 2: spatial-temporal convolution for video tokens.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the VideoLLaMA2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class VideoLLaMA2Options : VideoLanguageOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VideoLLaMA2Options(VideoLLaMA2Options other)
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
        MaxFrames = other.MaxFrames;
        LanguageModelName = other.LanguageModelName;
        ProjectionDim = other.ProjectionDim;
        SystemPrompt = other.SystemPrompt;
        EnableSpatialTemporalConv = other.EnableSpatialTemporalConv;
        VisionEncoderName = other.VisionEncoderName;
        PatchSize = other.PatchSize;
        VisionNumHeads = other.VisionNumHeads;
        DecoderNumHeads = other.DecoderNumHeads;
        DecoderNumKeyValueHeads = other.DecoderNumKeyValueHeads;
        VisionFfnDim = other.VisionFfnDim;
        DecoderFfnDim = other.DecoderFfnDim;
        RoPETheta = other.RoPETheta;
        STCKernelSize = other.STCKernelSize;
        STCStride = other.STCStride;
        STCPadding = other.STCPadding;
        STCStageDepth = other.STCStageDepth;
        STCMlpDepth = other.STCMlpDepth;
    }

    public VideoLLaMA2Options()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        VisionNumHeads = 16;
        DecoderNumHeads = 32;
        DecoderNumKeyValueHeads = 8;
        VisionFfnDim = 4096;
        DecoderFfnDim = 14336;
        ImageSize = 336;
        PatchSize = 14;
        VocabSize = 32000;
        MaxSequenceLength = 2048;
        DropoutRate = 0.0;
        LanguageModelName = "Mistral-7B-Instruct-v0.2";
        VisionEncoderName = "openai/clip-vit-large-patch14-336";
        MaxFrames = 8;
        LearningRate = 1e-3;
        WeightDecay = 0.0;
    }

    /// <summary>Gets or sets whether to use spatial-temporal convolution for video token compression.</summary>
    public bool EnableSpatialTemporalConv { get; set; } = true;

    /// <summary>Gets or sets the vision encoder identifier. The paper default is CLIP ViT-L/14 at 336px.</summary>
    public string VisionEncoderName { get; set; } = "openai/clip-vit-large-patch14-336";

    /// <summary>Gets or sets the square vision patch size. CLIP ViT-L/14 uses 14.</summary>
    public int PatchSize { get; set; } = 14;

    /// <summary>Gets or sets the number of CLIP vision-attention heads.</summary>
    public int VisionNumHeads { get; set; } = 16;

    /// <summary>Gets or sets the number of language-decoder query heads.</summary>
    public int DecoderNumHeads { get; set; } = 32;

    /// <summary>Gets or sets the number of language-decoder key/value heads.</summary>
    public int DecoderNumKeyValueHeads { get; set; } = 8;

    /// <summary>Gets or sets the CLIP vision-transformer feed-forward width.</summary>
    public int VisionFfnDim { get; set; } = 4096;

    /// <summary>Gets or sets the Mistral decoder SwiGLU intermediate width.</summary>
    public int DecoderFfnDim { get; set; } = 14336;

    /// <summary>Gets or sets the Mistral rotary-position base frequency.</summary>
    public double RoPETheta { get; set; } = 10000.0;

    /// <summary>Gets or sets the uniform temporal/spatial STC Conv3D kernel size.</summary>
    public int STCKernelSize { get; set; } = 2;

    /// <summary>Gets or sets the uniform temporal/spatial STC Conv3D stride.</summary>
    public int STCStride { get; set; } = 2;

    /// <summary>Gets or sets the uniform temporal/spatial STC Conv3D padding.</summary>
    public int STCPadding { get; set; } = 1;

    /// <summary>Gets or sets the number of RegNet blocks in each STC spatial stage.</summary>
    public int STCStageDepth { get; set; } = 4;

    /// <summary>Gets or sets the number of linear layers in the STC readout MLP.</summary>
    public int STCMlpDepth { get; set; } = 2;
}
