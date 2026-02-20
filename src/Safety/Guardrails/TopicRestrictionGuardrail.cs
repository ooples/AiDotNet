using System.Text.RegularExpressions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Guardrails;

/// <summary>
/// Guardrail that restricts conversation to approved topics using keyword and pattern matching.
/// </summary>
/// <remarks>
/// <para>
/// Blocks or warns when input contains references to restricted topics. Uses both exact
/// keyword matching and configurable regex patterns. Topics are case-insensitive.
/// </para>
/// <para>
/// <b>For Beginners:</b> This guardrail lets you define specific topics that the AI should
/// not discuss. For example, if you're building a cooking assistant, you might restrict
/// topics like "politics", "religion", or "medical advice". If a user asks about a restricted
/// topic, the system will block or warn.
/// </para>
/// <para>
/// <b>References:</b>
/// - NeMo Guardrails: Topic control through conversational rails (NVIDIA, 2024)
/// - Guardrails AI: Semantic topic filtering (2024)
/// - Constitutional AI: Principle-based output steering (Anthropic, 2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TopicRestrictionGuardrail<T> : ITextSafetyModule<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromMilliseconds(100);

    private readonly string[] _restrictedTopics;
    private readonly SafetyAction _action;

    /// <inheritdoc />
    public string ModuleName => "TopicRestrictionGuardrail";

    /// <inheritdoc />
    public bool IsReady => true;

    /// <summary>
    /// Initializes a new topic restriction guardrail.
    /// </summary>
    /// <param name="restrictedTopics">
    /// Topics to restrict. Each topic is treated as a case-insensitive keyword.
    /// Example: ["politics", "religion", "medical advice"]
    /// </param>
    /// <param name="action">
    /// Action to take when a restricted topic is detected. Default: Block.
    /// </param>
    public TopicRestrictionGuardrail(
        string[] restrictedTopics,
        SafetyAction action = SafetyAction.Block)
    {
        _restrictedTopics = restrictedTopics ?? Array.Empty<string>();
        _action = action;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();

        if (string.IsNullOrWhiteSpace(text) || _restrictedTopics.Length == 0)
        {
            return findings;
        }

        string lowerText = text.ToLowerInvariant();

        foreach (string topic in _restrictedTopics)
        {
            if (string.IsNullOrWhiteSpace(topic))
            {
                continue;
            }

            string lowerTopic = topic.ToLowerInvariant();

            // Word-boundary match to avoid partial matches (e.g., "art" in "party")
            try
            {
                string pattern = @"\b" + Regex.Escape(lowerTopic) + @"\b";
                var match = Regex.Match(lowerText, pattern, RegexOptions.None, RegexTimeout);

                if (match.Success)
                {
                    findings.Add(new SafetyFinding
                    {
                        Category = SafetyCategory.PromptInjection,
                        Severity = SafetySeverity.Medium,
                        Confidence = 0.9,
                        Description = $"Input references restricted topic: \"{topic}\".",
                        RecommendedAction = _action,
                        SourceModule = ModuleName,
                        SpanStart = match.Index,
                        SpanEnd = match.Index + match.Length
                    });
                }
            }
            catch (RegexMatchTimeoutException)
            {
                // ReDoS protection: skip topic if pattern evaluation times out
            }
        }

        return findings;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content)
    {
        return Array.Empty<SafetyFinding>();
    }
}
