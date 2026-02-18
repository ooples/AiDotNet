namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for RADIOv2.5, NVIDIA's agglomerative vision foundation model.
/// </summary>
/// <remarks>
/// <para>RADIOv2.5 (Ranzinger et al., 2025) distills multiple teacher models (DINOv2, SAM, SigLIP, CLIP)
/// into a single student model via multi-teacher distillation. The resulting model produces features
/// compatible with all teacher models, serving as a universal vision backbone.</para>
/// </remarks>
public class RADIOv25Options : VisionEncoderOptions
{
    /// <summary>
    /// Gets or sets the teacher models used for distillation.
    /// </summary>
    public string[] TeacherModels { get; set; } = ["DINOv2", "SAM", "SigLIP", "CLIP"];

    /// <summary>
    /// Gets or sets the number of summary tokens for compact representation.
    /// </summary>
    public int NumSummaryTokens { get; set; } = 1;

    /// <summary>
    /// Gets or sets whether to produce teacher-specific output heads.
    /// </summary>
    public bool UseTeacherSpecificHeads { get; set; } = true;

    /// <summary>
    /// Gets or sets the adapter dimension for teacher-specific heads.
    /// </summary>
    public int AdapterDim { get; set; } = 1280;

    public RADIOv25Options()
    {
        ImageSize = 432;
        EmbeddingDim = 1280;
        PatchSize = 16;
        NumLayers = 32;
        NumHeads = 16;
        ImageMean = [0.485, 0.456, 0.406];
        ImageStd = [0.229, 0.224, 0.225];
    }
}
