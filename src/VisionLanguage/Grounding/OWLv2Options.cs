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
        NumClassEmbeddings = other.NumClassEmbeddings;
        EnableSelfTraining = other.EnableSelfTraining;
    }

    public OWLv2Options()
    {
        VisionDim = 1024;
        DecoderDim = 1024;
        NumVisionLayers = 24;
        NumDecoderLayers = 6;
        NumHeads = 16;
        ImageSize = 960;
        MaxDetections = 100;
    }

    public int NumClassEmbeddings { get; set; } = 768;

    /// <summary>Gets or sets the text-encoder feature width.</summary>
    public int TextEmbeddingDim { get; set; } = 768;

    /// <summary>Gets or sets the detection-decoder feature width.</summary>
    public int DetectionDim { get; set; } = 256;

    /// <summary>Gets or sets the number of cross-modal fusion layers.</summary>
    public int NumFusionLayers { get; set; } = 6;

    /// <summary>Gets or sets whether self-training augmentation is enabled.</summary>
    public bool EnableSelfTraining { get; set; } = true;
}
