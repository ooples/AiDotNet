namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Configuration options for OWLv2: self-training for scaling open-vocabulary detection.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the OWLv2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class OWLv2Options : GroundingVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public OWLv2Options(OWLv2Options other)
        : base(other)
    {
        TextEmbeddingDim = other.TextEmbeddingDim;
        DetectionDim = other.DetectionDim;
        NumFusionLayers = other.NumFusionLayers;
        NumTextLayers = other.NumTextLayers;
        NumClassEmbeddings = other.NumClassEmbeddings;
        EnableSelfTraining = other.EnableSelfTraining;
        PatchSize = other.PatchSize;
    }

    /// <summary>Initializes a new instance with default values.</summary>
    public OWLv2Options()
    {
        VisionDim = 1024;
        DecoderDim = 1024;
        NumVisionLayers = 24;
        NumDecoderLayers = 6;
        NumHeads = 16;
        // OWLv2 L/14 runs at 1008 x 1008 (72 x 72 patches); B/16 runs at 960.
        ImageSize = 1008;
        MaxDetections = 100;
    }

    /// <summary>Gets or sets the width of the class-embedding head of the reference configuration. It is only recorded in the model metadata.</summary>
    public int NumClassEmbeddings { get; set; } = 768;

    /// <summary>Gets or sets the text-encoder feature width.</summary>
    public int TextEmbeddingDim { get; set; } = 768;

    /// <summary>Gets or sets the detection-decoder feature width.</summary>
    public int DetectionDim { get; set; } = 256;

    /// <summary>Gets or sets the number of cross-modal fusion layers.</summary>
    public int NumFusionLayers { get; set; } = 6;

    /// <summary>Gets or sets the number of layers of the CLIP text transformer that embeds the text queries (12).</summary>
    public int NumTextLayers { get; set; } = 12;

    /// <summary>
    /// Gets or sets the side of the square image patches the ViT backbone embeds (14 for the ViT-L/14 backbone; 16 for
    /// B/16). An image of side ImageSize becomes a square grid of
    /// (ImageSize / PatchSize)^2 patch tokens.
    /// </summary>
    public int PatchSize { get; set; } = 14;

    /// <summary>Gets or sets whether self-training augmentation is enabled.</summary>
    public bool EnableSelfTraining { get; set; } = true;
}
