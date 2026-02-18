namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for interactive, promptable segmentation models like SAM that accept
/// user prompts (points, boxes, masks, text) to segment specific objects.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Promptable segmentation models first encode an image, then accept various types of prompts
/// to segment specific regions. This two-stage design allows efficient interactive use where
/// the image is encoded once and multiple prompts can be processed quickly.
/// </para>
/// <para>
/// <b>For Beginners:</b> Promptable segmentation is like pointing at something in a photo
/// and having the model outline it for you.
///
/// Prompt types:
/// - Points: Click on an object and the model segments it
/// - Boxes: Draw a rectangle around an object for more precise segmentation
/// - Masks: Provide a rough mask and the model refines it
/// - Text: Describe what to segment (for models that support it)
///
/// Models implementing this interface:
/// - SAM / SAM 2 (Meta, foundation model for segmentation)
/// - SAM-HQ (high-quality boundaries)
/// - SegGPT (in-context learning)
/// - SEEM (multi-modal prompts including audio)
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("PromptableSegmentation")]
public interface IPromptableSegmentation<T> : ISegmentationModel<T>
{
    /// <summary>
    /// Gets whether this model supports point prompts.
    /// </summary>
    bool SupportsPointPrompts { get; }

    /// <summary>
    /// Gets whether this model supports box prompts.
    /// </summary>
    bool SupportsBoxPrompts { get; }

    /// <summary>
    /// Gets whether this model supports mask prompts.
    /// </summary>
    bool SupportsMaskPrompts { get; }

    /// <summary>
    /// Gets whether this model supports text prompts.
    /// </summary>
    bool SupportsTextPrompts { get; }

    /// <summary>
    /// Encodes an image for subsequent prompted segmentation.
    /// </summary>
    /// <param name="image">Input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Call this once per image. The model processes the image and
    /// stores the result internally. After this, you can call the prompt methods many times
    /// without re-processing the image, making interactive use very fast.
    /// </para>
    /// </remarks>
    void SetImage(Tensor<T> image);

    /// <summary>
    /// Segments a region indicated by point prompts.
    /// </summary>
    /// <param name="points">Point coordinates as [N, 2] tensor (x, y pairs).</param>
    /// <param name="labels">Point labels as [N] tensor (1 = foreground, 0 = background).</param>
    /// <returns>Predicted mask(s) with confidence scores.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Click on the object you want to segment (foreground point, label=1)
    /// and optionally click on areas that should NOT be included (background point, label=0).
    /// The model returns one or more mask proposals with confidence scores.
    /// </para>
    /// </remarks>
    PromptedSegmentationResult<T> SegmentFromPoints(Tensor<T> points, Tensor<T> labels);

    /// <summary>
    /// Segments a region indicated by a bounding box prompt.
    /// </summary>
    /// <param name="box">Bounding box as [4] tensor (x1, y1, x2, y2) in pixel coordinates.</param>
    /// <returns>Predicted mask(s) with confidence scores.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Draw a rectangle around the object you want to segment.
    /// The model returns a precise mask of the object within the box.
    /// </para>
    /// </remarks>
    PromptedSegmentationResult<T> SegmentFromBox(Tensor<T> box);

    /// <summary>
    /// Segments a region using a rough mask as a prompt.
    /// </summary>
    /// <param name="mask">Rough mask tensor [H, W] where positive values indicate the region of interest.</param>
    /// <returns>Refined mask(s) with confidence scores.</returns>
    PromptedSegmentationResult<T> SegmentFromMask(Tensor<T> mask);

    /// <summary>
    /// Generates masks for the entire image without any prompts (automatic mode).
    /// </summary>
    /// <returns>All detected segments in the image.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The model automatically finds and segments every distinct
    /// object or region in the image without any user input. This is useful for
    /// creating a complete segmentation of the scene.
    /// </para>
    /// </remarks>
    List<PromptedSegmentationResult<T>> SegmentEverything();
}

/// <summary>
/// Result from a prompted segmentation operation containing one or more mask proposals.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class PromptedSegmentationResult<T>
{
    /// <summary>
    /// Predicted binary masks [numMasks, H, W]. Multiple masks represent different possible
    /// interpretations of the prompt (e.g., whole object vs. part).
    /// </summary>
    public Tensor<T> Masks { get; set; } = Tensor<T>.Empty();

    /// <summary>
    /// Confidence scores for each mask proposal.
    /// </summary>
    public double[] Scores { get; set; } = [];

    /// <summary>
    /// Low-resolution mask logits that can be fed back as mask prompts for iterative refinement.
    /// </summary>
    public Tensor<T> LowResLogits { get; set; } = Tensor<T>.Empty();

    /// <summary>
    /// Stability scores measuring how consistent each mask is under small perturbations.
    /// </summary>
    public double[] StabilityScores { get; set; } = [];
}
