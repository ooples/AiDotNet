using AiDotNet.Enums;

namespace AiDotNet.Safety.Guardrails;

/// <summary>
/// Defines a declarative guardrail rule with a condition expression and corresponding action.
/// </summary>
/// <remarks>
/// <para>
/// A guardrail rule pairs a condition (what to detect) with an action (what to do about it).
/// Rules can check for regex patterns, keyword lists, length limits, or custom predicates.
/// Unlike <see cref="CustomRule"/> which uses raw delegates, GuardrailRule uses a structured
/// format that enables serialization, logging, and composition.
/// </para>
/// <para>
/// <b>For Beginners:</b> A rule is like an if-then statement for safety. The condition says
/// "if the text contains X" and the action says "then do Y (block, warn, log, or modify)."
/// You can build complex safety policies by combining multiple rules.
/// </para>
/// <para>
/// <b>References:</b>
/// - Guardrails AI: Structured validators (2024)
/// - NeMo Guardrails: Colang rule language (NVIDIA, 2024)
/// </para>
/// </remarks>
public class GuardrailRule
{
    /// <summary>
    /// Gets or sets the unique name for this rule.
    /// </summary>
    public string Name { get; set; } = "";

    /// <summary>
    /// Gets or sets the human-readable description of what this rule checks.
    /// </summary>
    public string Description { get; set; } = "";

    /// <summary>
    /// Gets or sets the type of condition this rule uses.
    /// </summary>
    public GuardrailConditionType ConditionType { get; set; } = GuardrailConditionType.ContainsAny;

    /// <summary>
    /// Gets or sets the patterns or keywords to check for (interpretation depends on ConditionType).
    /// For <see cref="GuardrailConditionType.ContainsAny"/> or <see cref="GuardrailConditionType.ContainsAll"/>: list of keywords.
    /// For <see cref="GuardrailConditionType.RegexMatch"/>: single regex pattern.
    /// For <see cref="GuardrailConditionType.MaxLength"/>: single string with the max length number.
    /// </summary>
    public string[] Patterns { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets the action to take when the condition is met.
    /// </summary>
    public SafetyAction Action { get; set; } = SafetyAction.Warn;

    /// <summary>
    /// Gets or sets the severity to assign when the rule triggers.
    /// </summary>
    public SafetySeverity Severity { get; set; } = SafetySeverity.Medium;

    /// <summary>
    /// Gets or sets the safety category to assign when the rule triggers.
    /// </summary>
    public SafetyCategory Category { get; set; } = SafetyCategory.PromptInjection;

    /// <summary>
    /// Gets or sets the direction this rule applies to.
    /// </summary>
    public GuardrailDirection Direction { get; set; } = GuardrailDirection.Both;

    /// <summary>
    /// Gets or sets whether pattern matching is case-insensitive. Default: true.
    /// </summary>
    public bool CaseInsensitive { get; set; } = true;

    /// <summary>
    /// Gets or sets an optional custom predicate for <see cref="GuardrailConditionType.Custom"/> rules.
    /// </summary>
    public Func<string, bool>? CustomPredicate { get; set; }

    /// <summary>
    /// Evaluates this rule against the given text.
    /// </summary>
    /// <param name="text">The text to evaluate.</param>
    /// <returns>True if the rule condition is met (violation detected); false otherwise.</returns>
    public bool Evaluate(string text)
    {
        if (string.IsNullOrEmpty(text)) return false;

        switch (ConditionType)
        {
            case GuardrailConditionType.ContainsAny:
                return EvaluateContainsAny(text);

            case GuardrailConditionType.ContainsAll:
                return EvaluateContainsAll(text);

            case GuardrailConditionType.RegexMatch:
                return EvaluateRegex(text);

            case GuardrailConditionType.MaxLength:
                return EvaluateMaxLength(text);

            case GuardrailConditionType.MinLength:
                return EvaluateMinLength(text);

            case GuardrailConditionType.Custom:
                return CustomPredicate?.Invoke(text) ?? false;

            default:
                return false;
        }
    }

    private bool EvaluateContainsAny(string text)
    {
        var comparison = CaseInsensitive ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        foreach (string pattern in Patterns)
        {
            if (text.Contains(pattern, comparison)) return true;
        }
        return false;
    }

    private bool EvaluateContainsAll(string text)
    {
        var comparison = CaseInsensitive ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        foreach (string pattern in Patterns)
        {
            if (!text.Contains(pattern, comparison)) return false;
        }
        return Patterns.Length > 0;
    }

    private bool EvaluateRegex(string text)
    {
        if (Patterns.Length == 0) return false;
        try
        {
            var options = CaseInsensitive
                ? System.Text.RegularExpressions.RegexOptions.IgnoreCase
                : System.Text.RegularExpressions.RegexOptions.None;
            return System.Text.RegularExpressions.Regex.IsMatch(
                text, Patterns[0], options, TimeSpan.FromMilliseconds(100));
        }
        catch
        {
            return false;
        }
    }

    private bool EvaluateMaxLength(string text)
    {
        if (Patterns.Length == 0) return false;
        if (int.TryParse(Patterns[0], out int maxLength))
        {
            return text.Length > maxLength;
        }
        return false;
    }

    private bool EvaluateMinLength(string text)
    {
        if (Patterns.Length == 0) return false;
        if (int.TryParse(Patterns[0], out int minLength))
        {
            return text.Length < minLength;
        }
        return false;
    }
}

/// <summary>
/// Specifies the type of condition a guardrail rule checks.
/// </summary>
public enum GuardrailConditionType
{
    /// <summary>Text contains any of the specified patterns.</summary>
    ContainsAny,

    /// <summary>Text contains all of the specified patterns.</summary>
    ContainsAll,

    /// <summary>Text matches a regular expression pattern.</summary>
    RegexMatch,

    /// <summary>Text exceeds a maximum character length.</summary>
    MaxLength,

    /// <summary>Text is shorter than a minimum character length.</summary>
    MinLength,

    /// <summary>Custom predicate function evaluates to true.</summary>
    Custom
}
