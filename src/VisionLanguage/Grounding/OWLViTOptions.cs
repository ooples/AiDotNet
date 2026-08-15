namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Configuration options for OWL-ViT: open-vocabulary object detection via ViT + CLIP alignment.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the OWLViT model. Default values follow the original paper settings.</para>
/// </remarks>
public class OWLViTOptions : GroundingVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public OWLViTOptions(OWLViTOptions other)
        : base(other)
    {
        TextEmbeddingDim = other.TextEmbeddingDim;
        DetectionDim = other.DetectionDim;
        NumFusionLayers = other.NumFusionLayers;
        NumClassEmbeddings = other.NumClassEmbeddings;
    }

    public OWLViTOptions()
    {
        VisionDim = 768;
        DecoderDim = 768;
        NumVisionLayers = 12;
        NumDecoderLayers = 6;
        NumHeads = 12;
        ImageSize = 768;
        MaxDetections = 100;
    }

    /// <summary>Gets or sets the class embedding dimension.</summary>
    public int NumClassEmbeddings { get; set; } = 512;

    /// <summary>Gets or sets the text-encoder feature width.</summary>
    public int TextEmbeddingDim { get; set; } = 768;

    /// <summary>Gets or sets the detection-decoder feature width.</summary>
    public int DetectionDim { get; set; } = 256;

    /// <summary>Gets or sets the number of cross-modal fusion layers.</summary>
    public int NumFusionLayers { get; set; } = 6;
}
