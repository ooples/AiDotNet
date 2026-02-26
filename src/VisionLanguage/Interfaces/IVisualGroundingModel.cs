namespace AiDotNet.VisionLanguage.Interfaces;

/// <summary>
/// Interface for visual grounding models that localize objects or regions from natural language descriptions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Visual grounding models take an image and a text query and produce bounding boxes,
/// segmentation masks, or region proposals for the objects/regions described by the text.
/// Architectures include:
/// <list type="bullet">
/// <item>Grounding DINO: DINO detector + grounded pre-training for open-set detection</item>
/// <item>GLaMM: Pixel grounding with LMM generating text + segmentation masks</item>
/// <item>Ferret: Spatial-aware visual sampler for free-form region inputs</item>
/// <item>OWL-ViT/OWLv2: Open-vocabulary detection via CLIP-aligned ViT</item>
/// </list>
/// </para>
/// </remarks>
public interface IVisualGroundingModel<T> : IVisualEncoder<T>
{
    /// <summary>
    /// Grounds a text query in the image, producing bounding box coordinates.
    /// </summary>
    /// <param name="image">Image tensor in [channels, height, width] format.</param>
    /// <param name="textQuery">Natural language description of the object/region to localize.</param>
    /// <returns>Tensor containing bounding box coordinates [x1, y1, x2, y2, confidence, ...] for detected regions.</returns>
    Tensor<T> GroundText(Tensor<T> image, string textQuery);

    /// <summary>
    /// Detects objects in the image matching a set of category descriptions.
    /// </summary>
    /// <param name="image">Image tensor in [channels, height, width] format.</param>
    /// <param name="categories">List of category names or descriptions to detect.</param>
    /// <returns>Tensor containing detection results [x1, y1, x2, y2, classIdx, confidence, ...] per detection.</returns>
    Tensor<T> DetectObjects(Tensor<T> image, IReadOnlyList<string> categories);

    /// <summary>
    /// Gets the maximum number of detections the model can produce per image.
    /// </summary>
    int MaxDetections { get; }
}
