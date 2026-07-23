namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Output for open-vocabulary and referring segmentation models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Open-vocabulary segmentation lets you segment objects described by
/// arbitrary text. This output maps each text query to its corresponding mask(s) in the image.
/// Referring segmentation adds reasoning capability — the model can also explain what it found.
/// </para>
/// </remarks>
public class OpenVocabSegmentationOutput<T>
{
    /// <summary>
    /// Per-query masks [numQueries, H, W].
    /// </summary>
    public Tensor<T>? Masks { get; set; }

    /// <summary>
    /// Combined semantic map [H, W] where each pixel is assigned to the best-matching query.
    /// </summary>
    public Tensor<T>? SemanticMap { get; set; }

    /// <summary>
    /// Text queries that were used.
    /// </summary>
    public string[] Queries { get; set; } = [];

    /// <summary>
    /// Per-query confidence scores.
    /// </summary>
    public double[] Scores { get; set; } = [];

    /// <summary>
    /// Per-query bounding boxes [numQueries, 4] as (x1, y1, x2, y2).
    /// </summary>
    public Tensor<T>? BoundingBoxes { get; set; }

    /// <summary>
    /// Text-image similarity scores [numQueries] indicating how well each query matches the image.
    /// </summary>
    public double[]? SimilarityScores { get; set; }

    /// <summary>
    /// Model's textual response (for referring/reasoning segmentation models like LISA).
    /// </summary>
    public string? TextResponse { get; set; }

    /// <summary>
    /// Grounding tokens connecting text spans to image regions (for grounded models like GLaMM).
    /// </summary>
    public List<GroundingToken>? GroundingTokens { get; set; }

    /// <summary>
    /// Inference time.
    /// </summary>
    public TimeSpan InferenceTime { get; set; }
}

/// <summary>
/// A grounding token linking a text span to an image region.
/// </summary>
public class GroundingToken
{
    /// <summary>
    /// Text span that was grounded.
    /// </summary>
    public string Text { get; set; } = string.Empty;

    /// <summary>
    /// Start character index in the full text response.
    /// </summary>
    public int StartIndex { get; set; }

    /// <summary>
    /// End character index in the full text response.
    /// </summary>
    public int EndIndex { get; set; }

    /// <summary>
    /// Bounding box [x1, y1, x2, y2] of the grounded region.
    /// </summary>
    public double[]? BoundingBox { get; set; }

    /// <summary>
    /// Mask index in the output masks tensor (if a mask was generated).
    /// </summary>
    public int? MaskIndex { get; set; }

    /// <summary>
    /// Confidence of the grounding.
    /// </summary>
    public double Confidence { get; set; }
}
