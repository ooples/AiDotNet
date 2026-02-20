using AiDotNet.Enums;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Configuration for PII detection modules.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Use this to configure which types of personal information
/// to detect (emails, phone numbers, SSNs, etc.) and what to do when PII is found
/// (mask it, hash it, remove it, etc.).
/// </para>
/// </remarks>
public class PIIDetectorConfig
{
    /// <summary>PII categories to detect. Null = all categories.</summary>
    public string[]? Categories { get; set; }

    /// <summary>Redaction strategy to apply when PII is found. Default: Mask.</summary>
    public RedactionStrategy? Redaction { get; set; }

    /// <summary>Locale for PII pattern matching (e.g., "en-US", "de-DE"). Default: "en-US".</summary>
    public string? Locale { get; set; }

    /// <summary>Minimum confidence to report a PII match. Default: 0.5.</summary>
    public double? MinConfidence { get; set; }

    /// <summary>Custom regex patterns for additional PII types.</summary>
    public Dictionary<string, string>? CustomPatterns { get; set; }

    internal RedactionStrategy EffectiveRedaction => Redaction ?? RedactionStrategy.Mask;
    internal string EffectiveLocale => Locale ?? "en-US";
    internal double EffectiveMinConfidence => MinConfidence ?? 0.5;
}
