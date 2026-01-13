using System;

namespace AiDotNet.Document;

/// <summary>
/// Represents the result of document question answering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> When you ask a question about a document, the model returns
/// an answer along with confidence information and optionally the evidence (where in
/// the document the answer came from).
/// </para>
/// </remarks>
public class DocumentQAResult<T>
{
    /// <summary>
    /// Gets the sentinel string used when a model has no answer.
    /// </summary>
    public static string NoAnswerSentinel { get; } = "[No answer found]";

    /// <summary>
    /// Gets the answer to the question.
    /// </summary>
    public string? Answer { get; init; }

    /// <summary>
    /// Gets the confidence score for the answer (0-1).
    /// </summary>
    public T Confidence { get; init; } = default!;

    /// <summary>
    /// Gets the confidence as a double value for comparison.
    /// </summary>
    public double ConfidenceValue { get; init; }

    /// <summary>
    /// Gets the original question that was asked.
    /// </summary>
    public string Question { get; init; } = string.Empty;

    /// <summary>
    /// Gets the evidence regions that support the answer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Evidence regions highlight where in the document the answer was found.
    /// This helps verify the answer and understand the model's reasoning.
    /// </para>
    /// </remarks>
    public IReadOnlyList<EvidenceRegion<T>>? Evidence { get; init; }

    /// <summary>
    /// Gets alternative answers if multiple were considered.
    /// </summary>
    public IReadOnlyList<AlternativeAnswer<T>>? AlternativeAnswers { get; init; }

    /// <summary>
    /// Gets the classification of confidence level.
    /// </summary>
    public ConfidenceLevel ConfidenceLevel
    {
        get
        {
            return ConfidenceValue switch
            {
                < 0.25 => ConfidenceLevel.VeryLow,
                < 0.50 => ConfidenceLevel.Low,
                < 0.75 => ConfidenceLevel.Medium,
                < 0.90 => ConfidenceLevel.High,
                _ => ConfidenceLevel.VeryHigh
            };
        }
    }

    /// <summary>
    /// Gets whether the model was able to find an answer.
    /// </summary>
    public bool HasAnswer =>
        !string.IsNullOrWhiteSpace(Answer)
        && !string.Equals(Answer, NoAnswerSentinel, StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Gets processing time in milliseconds.
    /// </summary>
    public double ProcessingTimeMs { get; init; }
}

/// <summary>
/// Represents a region in the document that provides evidence for an answer.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class EvidenceRegion<T>
{
    /// <summary>
    /// Gets the text content of the evidence.
    /// </summary>
    public string Text { get; init; } = string.Empty;

    /// <summary>
    /// Gets the bounding box of the evidence region.
    /// </summary>
    public Vector<T> BoundingBox { get; init; } = Vector<T>.Empty();

    /// <summary>
    /// Gets how relevant this evidence is to the answer (0-1).
    /// </summary>
    public T Relevance { get; init; } = default!;

    /// <summary>
    /// Gets the page number where the evidence was found (0-indexed).
    /// </summary>
    public int PageIndex { get; init; }
}

/// <summary>
/// Represents an alternative answer with lower confidence.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class AlternativeAnswer<T>
{
    /// <summary>
    /// Gets the alternative answer text.
    /// </summary>
    public string Answer { get; init; } = string.Empty;

    /// <summary>
    /// Gets the confidence score for this alternative.
    /// </summary>
    public T Confidence { get; init; } = default!;
}
