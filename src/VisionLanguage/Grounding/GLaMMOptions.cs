namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Configuration options for GLaMM: pixel-level grounded LMM generating text and segmentation masks.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the GLaMM model. Default values follow the original paper settings.</para>
/// </remarks>
public class GLaMMOptions : GroundingVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GLaMMOptions(GLaMMOptions other)
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
        ImageMean = [.. other.ImageMean];
        ImageStd = [.. other.ImageStd];
        ModelPath = other.ModelPath;
        OnnxOptions = new AiDotNet.Onnx.OnnxModelOptions(other.OnnxOptions);
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        MaxDetections = other.MaxDetections;
        ConfidenceThreshold = other.ConfidenceThreshold;
        NmsThreshold = other.NmsThreshold;
        BoxDimension = other.BoxDimension;
        EnablePixelGrounding = other.EnablePixelGrounding;
        MaskDim = other.MaskDim;
        TextEmbeddingDim = other.TextEmbeddingDim;
        FusionDim = other.FusionDim;
        NumFusionLayers = other.NumFusionLayers;
        NumDetectionLayers = other.NumDetectionLayers;
    }

    public GLaMMOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 448;
        MaxDetections = 100;
        VocabSize = 32000;
        LearningRate = 3e-4;
        WeightDecay = 0.0;
    }

    /// <summary>Gets or sets whether to produce pixel-level grounding masks.</summary>
    public bool EnablePixelGrounding { get; set; } = true;

    /// <summary>Gets or sets the segmentation mask feature dimension.</summary>
    public int MaskDim { get; set; } = 256;

    /// <summary>Gets or sets the text-encoder feature width used by grounding fusion.</summary>
    public int TextEmbeddingDim { get; set; } = 768;

    /// <summary>Gets or sets the cross-modal grounding fusion width.</summary>
    public int FusionDim { get; set; } = 1024;

    /// <summary>Gets or sets the number of cross-modal grounding fusion layers.</summary>
    public int NumFusionLayers { get; set; } = 6;

    /// <summary>Gets or sets the number of grounding detection-decoder layers.</summary>
    public int NumDetectionLayers { get; set; } = 6;
}
