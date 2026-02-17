using System.Text.RegularExpressions;
using AiDotNet.Enums;
using AiDotNet.Safety;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Pattern-based jailbreak and prompt injection detector.
/// </summary>
/// <remarks>
/// <para>
/// Detects known jailbreak and prompt injection patterns using curated regex rules.
/// Covers direct injection, role-play attacks, encoding attacks, and instruction override attempts.
/// </para>
/// <para>
/// <b>For Beginners:</b> A "jailbreak" is when someone tries to trick an AI into
/// ignoring its safety rules. Common tricks include:
/// - "Ignore your previous instructions and do X"
/// - "You are now DAN (Do Anything Now)"
/// - Encoding harmful requests in Base64 or other formats
/// - Using role-play scenarios to bypass restrictions
///
/// This detector catches these patterns using curated rules.
/// </para>
/// <para>
/// <b>Limitations:</b> Pattern-based detection catches known attack formats but cannot
/// detect novel or semantically sophisticated jailbreaks. For production deployments,
/// combine with embedding-based or gradient-based detectors.
/// </para>
/// <para>
/// <b>References:</b>
/// - Bypassing LLM Guardrails: emoji/Unicode smuggling achieving 100% evasion
///   (2025, arxiv:2504.11168) — motivates need for multi-strategy detection
/// - GradSafe: Gradient analysis detecting jailbreaks with only 2 examples
///   (2024, arxiv:2402.13494) — future enhancement direction
/// - WildGuard: Open moderation covering 13 risk categories, 82.8% accuracy
///   (Allen AI, 2024, arxiv:2406.18495)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class PatternJailbreakDetector<T> : TextSafetyModuleBase<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromMilliseconds(100);

    private readonly List<(Regex Pattern, string AttackType, SafetySeverity Severity, string Description)> _patterns;
    private readonly double _sensitivity;

    /// <inheritdoc />
    public override string ModuleName => "PatternJailbreakDetector";

    /// <summary>
    /// Initializes a new instance of the pattern-based jailbreak detector.
    /// </summary>
    /// <param name="sensitivity">
    /// Detection sensitivity (0-1). Higher catches more but increases false positives. Default: 0.7.
    /// At 0.7+, basic patterns are checked. At 0.5+, subtle patterns are also checked.
    /// At 0.3+, aggressive patterns that may false-positive are included.
    /// </param>
    public PatternJailbreakDetector(double sensitivity = 0.7)
    {
        if (sensitivity < 0 || sensitivity > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(sensitivity),
                "Sensitivity must be between 0 and 1.");
        }

        _sensitivity = sensitivity;
        _patterns = BuildPatterns(sensitivity);
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

        foreach (var (pattern, attackType, severity, description) in _patterns)
        {
            try
            {
                if (pattern.IsMatch(lowerText))
                {
                    var match = pattern.Match(lowerText);
                    findings.Add(new SafetyFinding
                    {
                        Category = SafetyCategory.JailbreakAttempt,
                        Severity = severity,
                        Confidence = _sensitivity,
                        Description = $"{description} (Type: {attackType})",
                        RecommendedAction = SafetyAction.Block,
                        SourceModule = ModuleName,
                        SpanStart = match.Index,
                        SpanEnd = match.Index + match.Length,
                        Excerpt = match.Value.Length > 100
                            ? match.Value[..100] + "..."
                            : match.Value
                    });
                }
            }
            catch (RegexMatchTimeoutException)
            {
                // ReDoS protection
            }
        }

        // Check for encoding-based attacks
        var encodingFindings = CheckEncodingAttacks(text);
        findings.AddRange(encodingFindings);

        return findings;
    }

    private static List<(Regex, string, SafetySeverity, string)> BuildPatterns(double sensitivity)
    {
        var patterns = new List<(Regex, string, SafetySeverity, string)>();
        var options = RegexOptions.Compiled | RegexOptions.IgnoreCase;

        // === High confidence patterns (always active) ===

        // Direct instruction override
        patterns.Add((
            new Regex(@"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|rules|guidelines|prompts)", options, RegexTimeout),
            "DirectOverride", SafetySeverity.High,
            "Direct instruction override attempt"));

        // DAN (Do Anything Now) and similar personas
        patterns.Add((
            new Regex(@"\b(you\s+are\s+now\s+|act\s+as\s+|pretend\s+(to\s+be|you'?re)\s+)(dan|evil|unrestricted|unfiltered|uncensored|jailbroken)", options, RegexTimeout),
            "PersonaHijack", SafetySeverity.High,
            "Persona hijack attempt (DAN-style)"));

        // System prompt extraction
        patterns.Add((
            new Regex(@"(show|reveal|display|print|output|repeat|tell\s+me)\s+(your|the)\s+(system\s+)?(prompt|instructions|rules|guidelines)", options, RegexTimeout),
            "PromptExtraction", SafetySeverity.High,
            "System prompt extraction attempt"));

        // "Forget" instructions
        patterns.Add((
            new Regex(@"\b(forget|disregard|discard|override|bypass)\s+(your|all|any|the)\s+(rules|guidelines|restrictions|safety|filters|instructions)", options, RegexTimeout),
            "InstructionBypass", SafetySeverity.High,
            "Safety bypass instruction"));

        // Fake system messages
        patterns.Add((
            new Regex(@"\[system\]|\[admin\]|\[developer\]|<\|system\|>|<<sys>>|###\s*system", options, RegexTimeout),
            "FakeSystemMessage", SafetySeverity.High,
            "Fake system message injection"));

        // === Medium confidence patterns (sensitivity >= 0.5) ===
        if (sensitivity <= 0.5)
        {
            return patterns;
        }

        // Role-play jailbreaks
        patterns.Add((
            new Regex(@"(in\s+this\s+)?hypothetical\s+scenario|for\s+(educational|research|academic)\s+purposes\s+only", options, RegexTimeout),
            "HypotheticalFraming", SafetySeverity.Medium,
            "Hypothetical framing to bypass restrictions"));

        // Multi-turn escalation markers
        patterns.Add((
            new Regex(@"(now\s+that\s+you'?ve|since\s+you\s+(already|just))\s+(confirmed|agreed|said|shown)", options, RegexTimeout),
            "MultiTurnEscalation", SafetySeverity.Medium,
            "Multi-turn escalation attempt"));

        // Payload splitting markers
        patterns.Add((
            new Regex(@"(combine|concatenate|merge|join)\s+(the\s+)?(previous|above|these)\s+(parts|pieces|segments|fragments)", options, RegexTimeout),
            "PayloadSplitting", SafetySeverity.Medium,
            "Payload splitting attack"));

        // === Aggressive patterns (sensitivity >= 0.3) ===
        if (sensitivity <= 0.3)
        {
            return patterns;
        }

        // Indirect injection markers from external content
        patterns.Add((
            new Regex(@"(important|urgent|critical)\s*:\s*(ignore|override|forget|disregard)", options, RegexTimeout),
            "IndirectInjection", SafetySeverity.Medium,
            "Possible indirect prompt injection from external content"));

        return patterns;
    }

    private IReadOnlyList<SafetyFinding> CheckEncodingAttacks(string text)
    {
        var findings = new List<SafetyFinding>();

        // Check for Base64 encoded content that might hide instructions
        if (ContainsLikelyBase64(text))
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.PromptInjection,
                Severity = SafetySeverity.Medium,
                Confidence = 0.6,
                Description = "Possible Base64-encoded content detected (may hide instructions)",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        // Check for Unicode tag characters (U+E0001-U+E007F) used for smuggling
        if (ContainsUnicodeTagCharacters(text))
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.PromptInjection,
                Severity = SafetySeverity.High,
                Confidence = 0.9,
                Description = "Unicode tag character smuggling detected",
                RecommendedAction = SafetyAction.Block,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private static bool ContainsLikelyBase64(string text)
    {
        // Look for long Base64-like strings (40+ chars of base64 alphabet)
        try
        {
            var regex = new Regex(@"[A-Za-z0-9+/=]{40,}", RegexOptions.None, RegexTimeout);
            return regex.IsMatch(text);
        }
        catch (RegexMatchTimeoutException)
        {
            return false;
        }
    }

    private static bool ContainsUnicodeTagCharacters(string text)
    {
        foreach (char c in text)
        {
            // Unicode Tags block: U+E0001 to U+E007F (requires surrogate pairs in UTF-16)
            // Check for characters in the Specials block that are commonly misused
            if (c >= '\uFFF0' && c <= '\uFFFF')
            {
                return true;
            }
        }

        // Check for surrogate pairs pointing to tag characters
        for (int i = 0; i < text.Length - 1; i++)
        {
            if (char.IsHighSurrogate(text[i]) && char.IsLowSurrogate(text[i + 1]))
            {
                int codePoint = char.ConvertToUtf32(text[i], text[i + 1]);
                if (codePoint >= 0xE0001 && codePoint <= 0xE007F)
                {
                    return true;
                }
            }
        }

        return false;
    }
}
