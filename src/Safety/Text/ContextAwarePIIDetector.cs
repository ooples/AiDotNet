using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Context-aware PII detector that reduces false positives by analyzing surrounding text.
/// </summary>
/// <remarks>
/// <para>
/// Wraps an inner PII detector and applies contextual validation to filter out false positives.
/// For example, "call me at 555-0100" is a phone number, but "HTTP error 404-0100" is not.
/// Analyzes a context window around each detection and uses heuristic rules to determine if
/// the detected pattern is actually PII in context.
/// </para>
/// <para>
/// <b>For Beginners:</b> Sometimes a number that looks like a phone number is actually a
/// product code, and something that looks like an email is a filename. This module uses
/// the words around each detection to figure out if it's really personal information.
/// </para>
/// <para>
/// <b>References:</b>
/// - CAPID: Context-aware PII detection reducing over-redaction in QA (2026, arxiv:2602.10074)
/// - HIPS method with GPT-4 for educational PII detection (2025, arxiv:2501.09765)
/// - False sense of privacy: surface-level PII removal insufficient (2025, arxiv:2504.21035)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ContextAwarePIIDetector<T> : TextSafetyModuleBase<T>
{
    private readonly TextSafetyModuleBase<T> _innerDetector;
    private readonly int _contextWindow;

    // Contexts that suggest a number is NOT a phone number
    private static readonly string[] PhoneFalsePositiveContexts =
    {
        "error", "code", "version", "http", "status", "port",
        "item", "product", "model", "serial", "reference", "ticket",
        "order", "invoice", "page", "line", "row", "id"
    };

    // Contexts that suggest an email pattern is NOT an email
    private static readonly string[] EmailFalsePositiveContexts =
    {
        "file://", "path:", "route:", "endpoint:", "example.com",
        "test@test", "noreply", "no-reply", "localhost"
    };

    /// <inheritdoc />
    public override string ModuleName => "ContextAwarePIIDetector";

    /// <summary>
    /// Initializes a new context-aware PII detector.
    /// </summary>
    /// <param name="innerDetector">The base PII detector to wrap. If null, uses RegexPIIDetector.</param>
    /// <param name="contextWindow">Number of characters to examine around each detection. Default: 50.</param>
    public ContextAwarePIIDetector(TextSafetyModuleBase<T>? innerDetector = null, int contextWindow = 50)
    {
        _innerDetector = innerDetector ?? new RegexPIIDetector<T>();
        _contextWindow = contextWindow;
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return Array.Empty<SafetyFinding>();
        }

        // Get raw detections from inner detector
        var rawFindings = _innerDetector.EvaluateText(text);
        var filteredFindings = new List<SafetyFinding>();
        string lower = text.ToLowerInvariant();

        foreach (var finding in rawFindings)
        {
            if (IsLikelyRealPII(finding, text, lower))
            {
                // Boost confidence slightly for context-validated findings
                double boostedConfidence = Math.Min(1.0, finding.Confidence * 1.1);
                filteredFindings.Add(new SafetyFinding
                {
                    Category = finding.Category,
                    Severity = finding.Severity,
                    Confidence = boostedConfidence,
                    Description = finding.Description + " [context-validated]",
                    RecommendedAction = finding.RecommendedAction,
                    SourceModule = ModuleName,
                    SpanStart = finding.SpanStart,
                    SpanEnd = finding.SpanEnd,
                    Excerpt = finding.Excerpt
                });
            }
        }

        return filteredFindings;
    }

    private bool IsLikelyRealPII(SafetyFinding finding, string text, string lower)
    {
        // Extract context around the detection
        int start = finding.SpanStart >= 0 ? finding.SpanStart : 0;
        int end = finding.SpanEnd >= 0 ? finding.SpanEnd : text.Length;

        int contextStart = Math.Max(0, start - _contextWindow);
        int contextEnd = Math.Min(text.Length, end + _contextWindow);

        string beforeContext = lower.Substring(contextStart, start - contextStart);
        string afterContext = lower.Substring(end, contextEnd - end);
        string fullContext = beforeContext + " " + afterContext;

        // Check for phone number false positives
        if (finding.Description.Contains("phone", StringComparison.OrdinalIgnoreCase) ||
            finding.Description.Contains("Phone", StringComparison.OrdinalIgnoreCase))
        {
            foreach (var fp in PhoneFalsePositiveContexts)
            {
                if (fullContext.Contains(fp, StringComparison.OrdinalIgnoreCase))
                {
                    return false;
                }
            }
        }

        // Check for email false positives
        if (finding.Description.Contains("email", StringComparison.OrdinalIgnoreCase) ||
            finding.Description.Contains("Email", StringComparison.OrdinalIgnoreCase))
        {
            foreach (var fp in EmailFalsePositiveContexts)
            {
                if (fullContext.Contains(fp, StringComparison.OrdinalIgnoreCase))
                {
                    return false;
                }
            }

            // Check if it's in a code block context
            if (beforeContext.Contains("```") || beforeContext.Contains("<code>"))
            {
                return false;
            }
        }

        // Check for SSN false positives (area codes, zip codes)
        if (finding.Description.Contains("SSN", StringComparison.OrdinalIgnoreCase))
        {
            if (fullContext.Contains("zip", StringComparison.OrdinalIgnoreCase) ||
                fullContext.Contains("area code", StringComparison.OrdinalIgnoreCase) ||
                fullContext.Contains("date", StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }
        }

        // Check for IP address false positives (version numbers)
        if (finding.Description.Contains("IP", StringComparison.OrdinalIgnoreCase))
        {
            if (fullContext.Contains("version", StringComparison.OrdinalIgnoreCase) ||
                fullContext.Contains("v.", StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }
        }

        return true; // Likely real PII
    }
}
