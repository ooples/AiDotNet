namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Represents a user prompt for interactive/promptable segmentation models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A prompt tells the model what you want to segment.
/// You can use different types of prompts:
/// - Points: Click on the object (foreground) or background
/// - Boxes: Draw a rectangle around the object
/// - Masks: Provide a rough outline for refinement
/// - Text: Describe what to segment in natural language
///
/// You can combine multiple prompt types for better results.
/// </para>
/// </remarks>
public class SegmentationPrompt<T>
{
    /// <summary>
    /// Point prompts as [N, 2] (x, y pairs in pixel coordinates).
    /// </summary>
    public Tensor<T>? Points { get; set; }

    /// <summary>
    /// Point labels: 1 = foreground, 0 = background, -1 = ambiguous.
    /// Must have the same length as the number of points.
    /// </summary>
    public int[]? PointLabels { get; set; }

    /// <summary>
    /// Box prompts as [N, 4] (x1, y1, x2, y2 per box in pixel coordinates).
    /// </summary>
    public Tensor<T>? Boxes { get; set; }

    /// <summary>
    /// Mask prompt [H, W] where positive values indicate foreground.
    /// Can be a rough mask from a previous iteration or user scribble.
    /// </summary>
    public Tensor<T>? MaskInput { get; set; }

    /// <summary>
    /// Text prompt for language-guided segmentation.
    /// </summary>
    public string? TextPrompt { get; set; }

    /// <summary>
    /// Negative text prompt describing what to exclude.
    /// </summary>
    public string? NegativeTextPrompt { get; set; }

    /// <summary>
    /// Audio prompt for multi-modal models (e.g., SEEM).
    /// </summary>
    public Tensor<T>? AudioPrompt { get; set; }

    /// <summary>
    /// Reference image for in-context learning (e.g., SegGPT).
    /// </summary>
    public Tensor<T>? ReferenceImage { get; set; }

    /// <summary>
    /// Reference mask corresponding to the reference image.
    /// </summary>
    public Tensor<T>? ReferenceMask { get; set; }

    /// <summary>
    /// Whether to return multiple mask proposals (SAM-style) or a single best mask.
    /// </summary>
    public bool ReturnMultipleMasks { get; set; } = true;

    /// <summary>
    /// Whether to return low-resolution logits for iterative refinement.
    /// </summary>
    public bool ReturnLowResLogits { get; set; }

    /// <summary>
    /// Target object IDs for tracking correction in video segmentation.
    /// </summary>
    public int[]? TargetObjectIds { get; set; }

    /// <summary>
    /// Type of the primary prompt being used.
    /// </summary>
    public PromptType PrimaryPromptType
    {
        get
        {
            if (TextPrompt != null) return PromptType.Text;
            if (Boxes != null) return PromptType.Box;
            if (Points != null) return PromptType.Point;
            if (MaskInput != null) return PromptType.Mask;
            if (AudioPrompt != null) return PromptType.Audio;
            if (ReferenceImage != null) return PromptType.Reference;
            return PromptType.None;
        }
    }
}

/// <summary>
/// Type of segmentation prompt.
/// </summary>
public enum PromptType
{
    /// <summary>No prompt (automatic mode).</summary>
    None,
    /// <summary>Point clicks on foreground/background.</summary>
    Point,
    /// <summary>Bounding box around the target.</summary>
    Box,
    /// <summary>Rough mask or scribble.</summary>
    Mask,
    /// <summary>Natural language description.</summary>
    Text,
    /// <summary>Audio signal for multi-modal models.</summary>
    Audio,
    /// <summary>Reference image+mask for in-context learning.</summary>
    Reference
}
