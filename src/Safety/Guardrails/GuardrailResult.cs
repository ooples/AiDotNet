using AiDotNet.Enums;

namespace AiDotNet.Safety.Guardrails;

/// <summary>
/// Result from guardrail evaluation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> GuardrailResult provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class GuardrailResult
{
    /// <summary>Whether the content passed the guardrail check.</summary>
    public bool Passed { get; init; }

    /// <summary>The action taken by the guardrail.</summary>
    public SafetyAction ActionTaken { get; init; }

    /// <summary>List of violated rules.</summary>
    public IReadOnlyList<string> ViolatedRules { get; init; } = Array.Empty<string>();

    /// <summary>The modified content, if the guardrail applied modifications.</summary>
    public string? ModifiedContent { get; init; }
}
