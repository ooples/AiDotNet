using System.Text.RegularExpressions;
using AiDotNet.Enums;
using AiDotNet.Safety;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Rule-based toxicity detector using pattern matching for harmful content detection.
/// </summary>
/// <remarks>
/// <para>
/// This detector uses curated regex patterns to identify toxic, hateful, violent, and
/// other harmful text content. It provides fast, deterministic detection without requiring
/// any ML model or external dependencies.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the simplest toxicity detector — it uses pattern matching
/// (like a word filter) to catch harmful content. It's fast and reliable for obvious cases
/// but may miss subtle or context-dependent toxicity. For production use with higher accuracy,
/// combine with an <c>EmbeddingToxicityDetector</c> or <c>ClassifierToxicityDetector</c>.
/// </para>
/// <para>
/// <b>Design Decisions:</b>
/// - Regex timeout of 100ms prevents ReDoS attacks
/// - Patterns use word boundaries (\b) to reduce false positives
/// - Each pattern is associated with a specific SafetyCategory for granular reporting
/// - Compiled regex for performance on repeated evaluations
/// </para>
/// <para>
/// <b>References:</b>
/// - GPT-3.5/Llama 2 achieving 80-90% accuracy in hate speech identification
///   (2024, arxiv:2403.08035) — rule-based provides baseline; ML provides higher accuracy
/// - MetaTox knowledge graph for enhanced LLM toxicity detection
///   (2024, arxiv:2412.15268) — knowledge-graph augmented approach for future enhancement
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class RuleBasedToxicityDetector<T> : TextSafetyModuleBase<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromMilliseconds(100);

    private readonly List<(Regex Pattern, SafetyCategory Category, SafetySeverity Severity, string Description)> _patterns;
    private readonly double _confidenceThreshold;

    /// <inheritdoc />
    public override string ModuleName => "RuleBasedToxicityDetector";

    /// <summary>
    /// Initializes a new instance of the rule-based toxicity detector.
    /// </summary>
    /// <param name="confidenceThreshold">
    /// Minimum confidence threshold for findings (0-1). Default: 0.7.
    /// Rule-based matches always have confidence 1.0, so this effectively
    /// acts as a filter for future scoring enhancements.
    /// </param>
    public RuleBasedToxicityDetector(double confidenceThreshold = 0.7)
    {
        if (confidenceThreshold < 0 || confidenceThreshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(confidenceThreshold),
                "Confidence threshold must be between 0 and 1.");
        }

        _confidenceThreshold = confidenceThreshold;
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
        var lowerText = text.ToLowerInvariant();

        foreach (var (pattern, category, severity, description) in _patterns)
        {
            try
            {
                var matches = pattern.Matches(lowerText);
                foreach (Match match in matches)
                {
                    findings.Add(new SafetyFinding
                    {
                        Category = category,
                        Severity = severity,
                        Confidence = 1.0,
                        Description = description,
                        RecommendedAction = severity >= SafetySeverity.High ? SafetyAction.Block : SafetyAction.Warn,
                        SourceModule = ModuleName,
                        SpanStart = match.Index,
                        SpanEnd = match.Index + match.Length,
                        Excerpt = match.Value
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

    private static List<(Regex, SafetyCategory, SafetySeverity, string)> BuildDefaultPatterns()
    {
        var patterns = new List<(Regex, SafetyCategory, SafetySeverity, string)>();
        var options = RegexOptions.Compiled | RegexOptions.IgnoreCase;

        // Violence
        patterns.Add((
            new Regex(@"\b(kill|murder|assassinate|slaughter|massacre)\s+(you|them|him|her|everyone)\b", options, RegexTimeout),
            SafetyCategory.ViolenceThreat, SafetySeverity.High,
            "Direct violence threat detected"));

        patterns.Add((
            new Regex(@"\b(bomb|explosive|detonate|blow\s+up)\b", options, RegexTimeout),
            SafetyCategory.ViolenceWeapons, SafetySeverity.High,
            "Weapons/explosives reference detected"));

        // Self-harm / suicide
        patterns.Add((
            new Regex(@"\b(how\s+to\s+(kill|harm)\s+(myself|yourself))\b", options, RegexTimeout),
            SafetyCategory.ViolenceSelfHarm, SafetySeverity.Critical,
            "Self-harm instruction request detected"));

        patterns.Add((
            new Regex(@"\b(commit\s+suicide|end\s+(my|your)\s+life|ways?\s+to\s+die)\b", options, RegexTimeout),
            SafetyCategory.ViolenceSuicide, SafetySeverity.Critical,
            "Suicide-related content detected"));

        // Hate speech
        patterns.Add((
            new Regex(@"\b(all\s+\w+\s+should\s+(die|be\s+killed))\b", options, RegexTimeout),
            SafetyCategory.HateSpeech, SafetySeverity.Critical,
            "Genocidal hate speech detected"));

        patterns.Add((
            new Regex(@"\b(racial\s+superiority|ethnic\s+cleansing|white\s+power|master\s+race)\b", options, RegexTimeout),
            SafetyCategory.HateSpeech, SafetySeverity.High,
            "Supremacist ideology detected"));

        // Dangerous instructions
        patterns.Add((
            new Regex(@"\b(how\s+to\s+(make|build|create)\s+(a\s+)?(bomb|weapon|explosive|poison))\b", options, RegexTimeout),
            SafetyCategory.WeaponsInstructions, SafetySeverity.Critical,
            "Weapons manufacturing instruction request detected"));

        patterns.Add((
            new Regex(@"\b(how\s+to\s+(make|cook|synthesize)\s+(meth|cocaine|heroin|fentanyl))\b", options, RegexTimeout),
            SafetyCategory.DrugManufacturing, SafetySeverity.Critical,
            "Drug manufacturing instruction request detected"));

        // Terrorism
        patterns.Add((
            new Regex(@"\b(join\s+(isis|al[\-\s]?qaeda|taliban)|jihad|caliphate|martyrdom\s+operation)\b", options, RegexTimeout),
            SafetyCategory.ViolenceTerrorism, SafetySeverity.Critical,
            "Terrorism-related content detected"));

        // CSAM (always critical, always block)
        patterns.Add((
            new Regex(@"\b(child\s+(porn|sex|nude)|underage\s+(sex|nude|porn)|minor\s+(sex|nude))\b", options, RegexTimeout),
            SafetyCategory.SexualMinors, SafetySeverity.Critical,
            "Child sexual abuse material reference detected"));

        // Social engineering / fraud
        patterns.Add((
            new Regex(@"\b(send\s+(me\s+)?(your|the)\s+(password|credit\s+card|ssn|social\s+security))\b", options, RegexTimeout),
            SafetyCategory.SocialEngineering, SafetySeverity.High,
            "Social engineering attempt detected"));

        return patterns;
    }
}
