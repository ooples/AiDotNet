using System.Globalization;
using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>One recorded edit a reasoning-model profile made to an outgoing chat request.</summary>
/// <remarks>
/// <para>
/// A reasoning model rejects settings that ordinary chat models accept, so the request that leaves the process is
/// not always the request the caller configured. Every such edit produces one of these records: which model was
/// targeted, which profile matched it, which setting changed, and how. Nothing here carries a credential, an
/// endpoint, or any prompt text — only the name of a setting and a short fixed explanation — so a diagnostic is
/// safe to log, serialize, or show to a user.
/// </para>
/// <para>
/// This is the piece the reference implementation has no equivalent of. Upstream builds a different parameter
/// dictionary for reasoning models and never records that <c>temperature</c> and <c>top_p</c> were omitted, so a
/// run configured with a temperature and pointed at an o-series model produces results the configuration cannot
/// explain. Here the omission is a first-class, inspectable fact.
/// </para>
/// <para><b>For Beginners:</b> Think of this as a receipt line. If the library had to remove your "creativity"
/// setting because the model you chose does not accept one, this is the note that says so. Collect these and you
/// can always answer the question "was the request I configured the request that was actually sent?"</para>
/// </remarks>
public sealed class ReasoningModelDiagnostic
{
    /// <summary>The longest message retained; a longer explanation is truncated to this many characters.</summary>
    public const int MaxMessageLength = 240;

    /// <summary>Initializes a diagnostic describing one request adjustment.</summary>
    /// <param name="modelId">The model the request was aimed at.</param>
    /// <param name="profileName">The reasoning profile that matched the model.</param>
    /// <param name="parameter">The setting that changed.</param>
    /// <param name="adjustment">How the setting changed.</param>
    /// <param name="message">A short explanation; <c>null</c> becomes an empty string.</param>
    /// <exception cref="ArgumentNullException"><paramref name="modelId"/> or <paramref name="profileName"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="modelId"/> or <paramref name="profileName"/> is empty or white space.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="parameter"/> or <paramref name="adjustment"/> is undefined.</exception>
    public ReasoningModelDiagnostic(
        string modelId,
        string profileName,
        ReasoningRequestParameter parameter,
        ReasoningParameterAdjustment adjustment,
        string? message = null)
    {
        Guard.NotNullOrWhiteSpace(modelId);
        Guard.NotNullOrWhiteSpace(profileName);
        if (!Enum.IsDefined(typeof(ReasoningRequestParameter), parameter))
        {
            throw new ArgumentOutOfRangeException(nameof(parameter), parameter, "Value must be a defined parameter.");
        }

        if (!Enum.IsDefined(typeof(ReasoningParameterAdjustment), adjustment))
        {
            throw new ArgumentOutOfRangeException(nameof(adjustment), adjustment, "Value must be a defined adjustment.");
        }

        ModelId = modelId.Trim();
        ProfileName = profileName.Trim();
        Parameter = parameter;
        Adjustment = adjustment;
        Message = message is null ? string.Empty : Bound(message);
    }

    /// <summary>Gets the model the request was aimed at.</summary>
    public string ModelId { get; }

    /// <summary>Gets the name of the reasoning profile that matched the model.</summary>
    public string ProfileName { get; }

    /// <summary>Gets the request setting that was changed.</summary>
    public ReasoningRequestParameter Parameter { get; }

    /// <summary>Gets how the setting was changed.</summary>
    public ReasoningParameterAdjustment Adjustment { get; }

    /// <summary>Gets the short explanation; empty when none was supplied.</summary>
    public string Message { get; }

    /// <summary>Gets the stable key identifying this kind of adjustment, used to collapse repeats.</summary>
    /// <remarks>
    /// One long run makes the same adjustment on every request. Grouping by this key keeps an in-memory summary
    /// bounded by the number of distinct settings rather than by the number of calls.
    /// </remarks>
    public string Key => ModelId + "|" + Parameter.ToString() + "|" + Adjustment.ToString();

    /// <summary>Returns a description that carries no prompt text, endpoint, or credential.</summary>
    /// <returns>The model, the profile, the setting, and the adjustment.</returns>
    public override string ToString() => string.Format(
        CultureInfo.InvariantCulture,
        "ReasoningModelDiagnostic({0}, {1}, {2}, {3})",
        ModelId,
        ProfileName,
        Parameter,
        Adjustment);

    private static string Bound(string text)
    {
        string collapsed = text.Replace('\r', ' ').Replace('\n', ' ').Trim();
        if (collapsed.Length <= MaxMessageLength) return collapsed;
        return collapsed.Substring(0, MaxMessageLength - 3) + "...";
    }
}
