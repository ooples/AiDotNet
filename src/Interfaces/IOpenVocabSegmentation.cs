namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for open-vocabulary segmentation models that segment objects from text descriptions
/// without being limited to a fixed set of classes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Open-vocabulary segmentation models use vision-language understanding (typically CLIP-based)
/// to segment objects described by arbitrary text. Unlike traditional models trained on a fixed
/// set of classes, these models can segment any concept expressible in natural language.
/// </para>
/// <para>
/// <b>For Beginners:</b> Traditional segmentation models can only recognize objects they were
/// trained on (e.g., "car", "person", "dog"). Open-vocabulary models can segment anything
/// you describe in words — even things they've never seen during training.
///
/// Example: You can ask for "red coffee mug on the table" or "person wearing a blue hat"
/// and the model will try to segment exactly that, even if it was never trained on those
/// specific combinations.
///
/// Models implementing this interface:
/// - SAN (CVPR 2023, side adapter on CLIP)
/// - CAT-Seg (CVPR 2024, cost aggregation)
/// - SED (CVPR 2024, simple encoder-decoder)
/// - Open-Vocabulary SAM (ECCV 2024, SAM + CLIP)
/// - Grounded SAM 2 (grounding DINO + SAM 2)
/// - Mask-Adapter (CVPR 2025, mask-level adaptation)
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("OpenVocabSegmentation")]
public interface IOpenVocabSegmentation<T> : ISegmentationModel<T>
{
    /// <summary>
    /// Gets the maximum number of text categories that can be queried simultaneously.
    /// </summary>
    int MaxCategories { get; }

    /// <summary>
    /// Gets the maximum text prompt length in tokens.
    /// </summary>
    int MaxPromptLength { get; }

    /// <summary>
    /// Segments an image using text descriptions of target classes.
    /// </summary>
    /// <param name="image">Input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <param name="classNames">List of text descriptions for classes to segment
    /// (e.g., ["car", "person on bicycle", "stop sign"]).</param>
    /// <returns>Open-vocabulary segmentation result with per-class masks and scores.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Give the model an image and a list of things to find.
    /// The model returns masks showing where each described thing is in the image.
    /// You can use any description — you're not limited to predefined categories.
    /// </para>
    /// </remarks>
    OpenVocabSegmentationResult<T> SegmentWithText(Tensor<T> image, IReadOnlyList<string> classNames);

    /// <summary>
    /// Segments an image using a single text prompt for grounded segmentation.
    /// </summary>
    /// <param name="image">Input image tensor [C, H, W].</param>
    /// <param name="prompt">Natural language description (e.g., "the largest dog in the image").</param>
    /// <returns>Segmentation result for the described object(s).</returns>
    OpenVocabSegmentationResult<T> SegmentWithPrompt(Tensor<T> image, string prompt);
}

/// <summary>
/// Result of open-vocabulary segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class OpenVocabSegmentationResult<T>
{
    /// <summary>
    /// Per-class masks [numClasses, H, W].
    /// </summary>
    public Tensor<T> Masks { get; set; } = Tensor<T>.Empty();

    /// <summary>
    /// Class names corresponding to each mask.
    /// </summary>
    public string[] ClassNames { get; set; } = [];

    /// <summary>
    /// Per-class confidence scores.
    /// </summary>
    public double[] Scores { get; set; } = [];

    /// <summary>
    /// Per-pixel semantic map [H, W] with class indices.
    /// </summary>
    public Tensor<T> SemanticMap { get; set; } = Tensor<T>.Empty();
}
