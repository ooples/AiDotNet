using System.Text.RegularExpressions;
using AiDotNet.Enums;
using AiDotNet.Safety;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Regex-based PII (Personally Identifiable Information) detector for common PII patterns.
/// </summary>
/// <remarks>
/// <para>
/// Detects personally identifiable information using curated regex patterns for common PII
/// types: email addresses, phone numbers, Social Security Numbers, credit card numbers,
/// IP addresses, API keys, and more.
/// </para>
/// <para>
/// <b>For Beginners:</b> This detector scans text for personal information that shouldn't
/// be shared — like email addresses, phone numbers, credit card numbers, and Social Security
/// Numbers. When found, it reports the location so the information can be redacted.
/// </para>
/// <para>
/// <b>Limitations:</b> Regex-based detection has high precision for structured PII (emails,
/// SSNs, credit cards) but cannot detect unstructured PII (names, addresses in free text).
/// For comprehensive PII detection, combine with a NER-based detector.
/// </para>
/// <para>
/// <b>References:</b>
/// - PRvL: LLMs for contextual PII redaction outperform rule-based NER (2025, arxiv:2508.05545)
/// - CAPID: Context-aware PII detection reducing over-redaction (2026, arxiv:2602.10074)
/// - Hybrid multilingual PII detection evaluation (2025, arxiv:2510.07551)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class RegexPIIDetector<T> : TextSafetyModuleBase<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromMilliseconds(100);

    private readonly List<(Regex Pattern, string PIIType, SafetySeverity Severity, string Description)> _patterns;

    /// <inheritdoc />
    public override string ModuleName => "RegexPIIDetector";

    /// <summary>
    /// Initializes a new instance of the regex-based PII detector.
    /// </summary>
    public RegexPIIDetector()
    {
        _patterns = BuildDefaultPatterns();
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return Array.Empty<SafetyFinding>();
        }

        var findings = new List<SafetyFinding>();

        foreach (var (pattern, piiType, severity, description) in _patterns)
        {
            try
            {
                var matches = pattern.Matches(text);
                foreach (Match match in matches)
                {
                    findings.Add(new SafetyFinding
                    {
                        Category = SafetyCategory.PIIExposure,
                        Severity = severity,
                        Confidence = 0.95,
                        Description = $"{description} ({piiType})",
                        RecommendedAction = SafetyAction.Modify,
                        SourceModule = ModuleName,
                        SpanStart = match.Index,
                        SpanEnd = match.Index + match.Length,
                        Excerpt = MaskPII(match.Value)
                    });
                }
            }
            catch (RegexMatchTimeoutException)
            {
                // ReDoS protection: skip this pattern if it times out
            }
        }

        return findings;
    }

    /// <summary>
    /// Masks PII for safe inclusion in findings (shows first/last char only).
    /// </summary>
    private static string MaskPII(string value)
    {
        if (value.Length <= 2)
        {
            return "***";
        }

        return value[0] + new string('*', value.Length - 2) + value[^1];
    }

    private static List<(Regex, string, SafetySeverity, string)> BuildDefaultPatterns()
    {
        var patterns = new List<(Regex, string, SafetySeverity, string)>();
        var options = RegexOptions.Compiled;

        // Email addresses
        patterns.Add((
            new Regex(@"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", options, RegexTimeout),
            "Email", SafetySeverity.Medium,
            "Email address detected"));

        // US Social Security Numbers (XXX-XX-XXXX)
        patterns.Add((
            new Regex(@"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b", options, RegexTimeout),
            "SSN", SafetySeverity.Critical,
            "Social Security Number detected"));

        // Credit card numbers (13-19 digits, optionally separated by spaces/dashes)
        patterns.Add((
            new Regex(@"\b(?:\d{4}[-\s]?){3,4}\d{1,4}\b", options, RegexTimeout),
            "CreditCard", SafetySeverity.Critical,
            "Credit card number detected"));

        // US phone numbers
        patterns.Add((
            new Regex(@"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", options, RegexTimeout),
            "Phone", SafetySeverity.Medium,
            "Phone number detected"));

        // IPv4 addresses
        patterns.Add((
            new Regex(@"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b", options, RegexTimeout),
            "IPAddress", SafetySeverity.Low,
            "IP address detected"));

        // API keys / tokens (long hex or base64 strings)
        patterns.Add((
            new Regex(@"\b(?:sk|pk|api|token|key|secret)[-_]?[a-zA-Z0-9]{20,}\b", options | RegexOptions.IgnoreCase, RegexTimeout),
            "APIKey", SafetySeverity.High,
            "API key or token detected"));

        // AWS access keys (AKIA...)
        patterns.Add((
            new Regex(@"\bAKIA[0-9A-Z]{16}\b", options, RegexTimeout),
            "AWSKey", SafetySeverity.Critical,
            "AWS access key detected"));

        // Passwords in common patterns
        patterns.Add((
            new Regex(@"(?:password|passwd|pwd)\s*[:=]\s*\S+", options | RegexOptions.IgnoreCase, RegexTimeout),
            "Password", SafetySeverity.Critical,
            "Password in plaintext detected"));

        // US Passport numbers (9 digits)
        patterns.Add((
            new Regex(@"\bpassport\s*(?:number|no|#)?\s*[:=]?\s*\d{9}\b", options | RegexOptions.IgnoreCase, RegexTimeout),
            "Passport", SafetySeverity.High,
            "Passport number detected"));

        // US Driver's license (varies by state, catch common formats)
        patterns.Add((
            new Regex(@"\b(?:driver'?s?\s*license|DL)\s*(?:number|no|#)?\s*[:=]?\s*[A-Z0-9]{5,15}\b", options | RegexOptions.IgnoreCase, RegexTimeout),
            "DriversLicense", SafetySeverity.High,
            "Driver's license number detected"));

        return patterns;
    }
}
