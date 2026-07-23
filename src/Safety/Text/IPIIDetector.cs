using AiDotNet.Interfaces;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Interface for PII (Personally Identifiable Information) detection modules.
/// </summary>
/// <remarks>
/// <para>
/// PII detectors identify sensitive personal information in text including names, emails,
/// phone numbers, SSNs, credit cards, addresses, API keys, and other data that could
/// compromise privacy if exposed.
/// </para>
/// <para>
/// <b>For Beginners:</b> A PII detector finds personal information like email addresses,
/// phone numbers, and social security numbers in text. This helps prevent accidentally
/// sharing private data through AI outputs.
/// </para>
/// <para>
/// <b>References:</b>
/// - PRvL: LLMs for contextual PII redaction outperform rule-based NER (2025, arxiv:2508.05545)
/// - CAPID: Context-aware PII detection reducing over-redaction (2026, arxiv:2602.10074)
/// - Hybrid multilingual PII detection (2025, arxiv:2510.07551)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IPIIDetector<T> : ITextSafetyModule<T>
{
    /// <summary>
    /// Detects PII entities in the given text and returns their locations and types.
    /// </summary>
    /// <param name="text">The text to scan for PII.</param>
    /// <returns>A list of detected PII entities with their spans, types, and confidence.</returns>
    IReadOnlyList<PIIEntity> DetectPII(string text);
}

/// <summary>
/// Represents a detected PII entity in text.
/// </summary>
public class PIIEntity
{
    /// <summary>The type of PII detected (e.g., "Email", "SSN", "PhoneNumber").</summary>
    public string Type { get; init; } = string.Empty;

    /// <summary>The matched text value.</summary>
    public string Value { get; init; } = string.Empty;

    /// <summary>Start character offset in the original text.</summary>
    public int StartIndex { get; init; }

    /// <summary>End character offset in the original text (exclusive, consistent with <see cref="SafetyFinding.SpanEnd"/>).</summary>
    public int EndIndex { get; init; }

    /// <summary>Detection confidence between 0.0 and 1.0.</summary>
    public double Confidence { get; init; }
}
