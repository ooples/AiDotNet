namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for referring segmentation models that segment objects based on natural language
/// descriptions, including complex reasoning about spatial relationships and attributes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Referring segmentation goes beyond open-vocabulary segmentation by supporting complex
/// natural language queries that require reasoning about object attributes, spatial relationships,
/// and even world knowledge. These models typically use large language models (LLMs) to
/// understand complex queries.
/// </para>
/// <para>
/// <b>For Beginners:</b> Referring segmentation lets you describe exactly what you want
/// using natural language — even complex descriptions.
///
/// Examples of what you can ask:
/// - "The person standing behind the counter" (spatial reasoning)
/// - "The animal that could be dangerous" (world knowledge / reasoning)
/// - "The object that doesn't belong in this kitchen" (contextual reasoning)
/// - "Track the person in the red shirt throughout the video" (video + language)
///
/// This is more powerful than open-vocabulary segmentation because it understands context
/// and relationships, not just category names.
///
/// Models implementing this interface:
/// - LISA (CVPR 2024 Oral, LLaVA + SAM reasoning segmentation)
/// - VideoLISA (NeurIPS 2024, video reasoning segmentation)
/// - GLaMM (CVPR 2024, grounded language model)
/// - OMG-LLaVA (NeurIPS 2024, pixel-level reasoning)
/// - PixelLM (CVPR 2024, segmentation codebook)
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("ReferringSegmentation")]
public interface IReferringSegmentation<T> : ISegmentationModel<T>
{
    /// <summary>
    /// Gets the maximum input text length in tokens.
    /// </summary>
    int MaxTextLength { get; }

    /// <summary>
    /// Gets whether this model supports multi-turn conversation.
    /// </summary>
    bool SupportsConversation { get; }

    /// <summary>
    /// Gets whether this model supports video input.
    /// </summary>
    bool SupportsVideoInput { get; }

    /// <summary>
    /// Segments objects described by a natural language expression.
    /// </summary>
    /// <param name="image">Input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <param name="expression">Natural language referring expression describing the target object(s).</param>
    /// <returns>Segmentation result with mask(s) and the model's textual response.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Describe what you want to segment in plain English.
    /// The model understands context, attributes, and spatial relationships.
    /// It returns both a mask and a text explanation of what it segmented.
    /// </para>
    /// </remarks>
    ReferringSegmentationResult<T> SegmentFromExpression(Tensor<T> image, string expression);

    /// <summary>
    /// Segments objects from a multi-turn conversation context.
    /// </summary>
    /// <param name="image">Input image tensor [C, H, W].</param>
    /// <param name="conversationHistory">Previous conversation turns as (role, message) pairs.</param>
    /// <param name="currentQuery">The current user query.</param>
    /// <returns>Segmentation result with mask(s) and conversational response.</returns>
    ReferringSegmentationResult<T> SegmentFromConversation(
        Tensor<T> image,
        IReadOnlyList<(string Role, string Message)> conversationHistory,
        string currentQuery);

    /// <summary>
    /// Segments objects in a video based on a natural language expression.
    /// </summary>
    /// <param name="frames">Video frames [numFrames, C, H, W].</param>
    /// <param name="expression">Natural language expression describing what to segment/track.</param>
    /// <returns>Per-frame segmentation results with tracking.</returns>
    List<ReferringSegmentationResult<T>> SegmentVideoFromExpression(Tensor<T> frames, string expression);
}

/// <summary>
/// Result of referring segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ReferringSegmentationResult<T>
{
    /// <summary>
    /// Binary mask(s) for the referred object(s) [numObjects, H, W].
    /// </summary>
    public Tensor<T> Masks { get; set; } = Tensor<T>.Empty();

    /// <summary>
    /// The model's textual response explaining what was segmented.
    /// </summary>
    public string TextResponse { get; set; } = string.Empty;

    /// <summary>
    /// Confidence score for the segmentation.
    /// </summary>
    public double Confidence { get; set; }

    /// <summary>
    /// Bounding boxes for each referred object [numObjects, 4] as (x1, y1, x2, y2).
    /// </summary>
    public Tensor<T> BoundingBoxes { get; set; } = Tensor<T>.Empty();

    /// <summary>
    /// Frame index (for video segmentation results).
    /// </summary>
    public int FrameIndex { get; set; }
}
