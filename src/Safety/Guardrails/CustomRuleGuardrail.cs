using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Guardrails;

/// <summary>
/// User-defined guardrail that applies custom validation rules to text content.
/// </summary>
/// <remarks>
/// <para>
/// Allows users to define arbitrary validation rules as delegates. Each rule receives
/// the input text and returns a SafetyFinding if the rule is violated. This provides
/// maximum flexibility for application-specific safety requirements.
/// </para>
/// <para>
/// <b>For Beginners:</b> This guardrail lets you define your own custom safety rules.
/// You write a function that checks the input and returns a safety finding if something
/// is wrong. This is useful for application-specific rules that the built-in modules
/// don't cover.
/// </para>
/// <para>
/// <b>Example usage:</b>
/// <code>
/// var customRule = new CustomRuleGuardrail&lt;double&gt;(new[]
/// {
///     new CustomRule("NoCompetitorMentions",
///         text => text.Contains("CompetitorName", StringComparison.OrdinalIgnoreCase)
///             ? new SafetyFinding { Description = "Competitor mentioned" }
///             : null)
/// });
/// </code>
/// </para>
/// <para>
/// <b>References:</b>
/// - Guardrails AI: Custom validators for production LLM systems (2024)
/// - NeMo Guardrails: Programmable rails (NVIDIA, 2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CustomRuleGuardrail<T> : ITextSafetyModule<T>
{
    private readonly IReadOnlyList<CustomRule> _rules;

    /// <inheritdoc />
    public string ModuleName => "CustomRuleGuardrail";

    /// <inheritdoc />
    public bool IsReady => true;

    /// <summary>
    /// Initializes a new custom rule guardrail.
    /// </summary>
    /// <param name="rules">The custom rules to apply.</param>
    public CustomRuleGuardrail(IReadOnlyList<CustomRule> rules)
    {
        _rules = rules ?? Array.Empty<CustomRule>();
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();

        if (string.IsNullOrEmpty(text))
        {
            return findings;
        }

        foreach (var rule in _rules)
        {
            try
            {
                var finding = rule.Evaluate(text);
                if (finding != null)
                {
                    // Create a new finding with source module set if not already specified
                    if (string.IsNullOrEmpty(finding.SourceModule))
                    {
                        finding = new SafetyFinding
                        {
                            Category = finding.Category,
                            Severity = finding.Severity,
                            Confidence = finding.Confidence,
                            Description = finding.Description,
                            RecommendedAction = finding.RecommendedAction,
                            SourceModule = $"{ModuleName}:{rule.Name}",
                            SpanStart = finding.SpanStart,
                            SpanEnd = finding.SpanEnd,
                            Excerpt = finding.Excerpt
                        };
                    }

                    findings.Add(finding);
                }
            }
            catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or FormatException or TimeoutException)
            {
                findings.Add(CreateRuleErrorFinding(rule.Name));
            }
        }

        return findings;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content)
    {
        if (content is null)
        {
            throw new ArgumentNullException(nameof(content));
        }

        // Convert vector to string (character codes) and delegate to text evaluation
        var numOps = MathHelper.GetNumericOperations<T>();
        var chars = new char[content.Length];
        for (int i = 0; i < content.Length; i++)
        {
            int code = (int)Math.Round(numOps.ToDouble(content[i]));
            chars[i] = code is >= 0 and <= 65535 ? (char)code : '?';
        }

        return EvaluateText(new string(chars));
    }

    private static SafetyFinding CreateRuleErrorFinding(string ruleName)
    {
        return new SafetyFinding
        {
            Category = SafetyCategory.PromptInjection,
            Severity = SafetySeverity.Low,
            Confidence = 0.5,
            Description = $"Custom rule '{ruleName}' failed during evaluation.",
            RecommendedAction = SafetyAction.Log,
            SourceModule = "CustomRuleGuardrail"
        };
    }
}

/// <summary>
/// A custom safety rule that evaluates text and optionally returns a finding.
/// </summary>
public class CustomRule
{
    private readonly Func<string, SafetyFinding?> _evaluator;

    /// <summary>
    /// Gets the name of this custom rule.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Creates a new custom rule.
    /// </summary>
    /// <param name="name">A descriptive name for the rule.</param>
    /// <param name="evaluator">
    /// A function that receives text and returns a SafetyFinding if the rule
    /// is violated, or null if the text passes the rule.
    /// </param>
    public CustomRule(string name, Func<string, SafetyFinding?> evaluator)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
        _evaluator = evaluator ?? throw new ArgumentNullException(nameof(evaluator));
    }

    /// <summary>
    /// Evaluates the text against this rule.
    /// </summary>
    /// <param name="text">The text to evaluate.</param>
    /// <returns>A finding if the rule is violated; null otherwise.</returns>
    public SafetyFinding? Evaluate(string text) => _evaluator(text);
}
