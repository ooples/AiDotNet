using AiDotNet.Enums;

namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>Converts a <see cref="ReasoningEffortLevel"/> to and from the value providers use on the wire.</summary>
/// <remarks>
/// <para>
/// Providers spell the deliberation level as a lower-case string in a <c>reasoning_effort</c> field. Keeping the
/// conversion in one place means a connector never hand-writes the literal, and
/// <see cref="ReasoningEffortLevel.Unspecified"/> reliably maps to "omit the field" rather than to some string the
/// provider would reject.
/// </para>
/// <para><b>For Beginners:</b> The library uses a readable name such as <c>High</c>; the provider wants the text
/// <c>"high"</c>. These two helpers translate between the two, and treat "unspecified" as "do not send anything".
/// </para>
/// </remarks>
public static class ReasoningEffortLevelExtensions
{
    /// <summary>Returns the wire value for an effort level, or <c>null</c> when no field should be sent.</summary>
    /// <param name="level">The effort level.</param>
    /// <returns>
    /// <c>minimal</c>, <c>low</c>, <c>medium</c>, or <c>high</c>; <c>null</c> for
    /// <see cref="ReasoningEffortLevel.Unspecified"/> and for any value outside the defined set.
    /// </returns>
    public static string? ToWireValue(this ReasoningEffortLevel level) => level switch
    {
        ReasoningEffortLevel.Minimal => "minimal",
        ReasoningEffortLevel.Low => "low",
        ReasoningEffortLevel.Medium => "medium",
        ReasoningEffortLevel.High => "high",
        _ => null
    };

    /// <summary>Parses a provider wire value into an effort level.</summary>
    /// <param name="value">The wire value; <c>null</c>, empty, or unrecognized yields <c>Unspecified</c>.</param>
    /// <returns>The matching level, or <see cref="ReasoningEffortLevel.Unspecified"/>.</returns>
    public static ReasoningEffortLevel ParseReasoningEffort(string? value)
    {
        // Narrowed by an explicit null check rather than string.IsNullOrWhiteSpace, which carries no nullable
        // annotation on .NET Framework and would leave the following Trim() flagged there.
        if (value is null) return ReasoningEffortLevel.Unspecified;
        string trimmed = value.Trim();
        if (trimmed.Length == 0) return ReasoningEffortLevel.Unspecified;
        if (string.Equals(trimmed, "minimal", StringComparison.OrdinalIgnoreCase)) return ReasoningEffortLevel.Minimal;
        if (string.Equals(trimmed, "low", StringComparison.OrdinalIgnoreCase)) return ReasoningEffortLevel.Low;
        if (string.Equals(trimmed, "medium", StringComparison.OrdinalIgnoreCase)) return ReasoningEffortLevel.Medium;
        if (string.Equals(trimmed, "high", StringComparison.OrdinalIgnoreCase)) return ReasoningEffortLevel.High;
        return ReasoningEffortLevel.Unspecified;
    }
}
