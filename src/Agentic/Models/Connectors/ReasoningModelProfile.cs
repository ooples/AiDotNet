using System.Collections.ObjectModel;
using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>Describes one family of reasoning models: how to recognise it, and how its requests differ.</summary>
/// <remarks>
/// <para>
/// Reasoning models are recognised by the start of their model id, not by the endpoint they are served from. That
/// is deliberate and matches the reference implementation: the same <c>o3</c> weights are reachable through
/// OpenAI, Azure OpenAI, OpenRouter, and local proxies, and all of them enforce the same parameter rules, so a
/// check on the base URL would be wrong on three of the four. A profile therefore carries a list of model-id
/// prefixes and is matched case-insensitively against <c>IChatClient&lt;T&gt;.ModelId</c>.
/// </para>
/// <para>
/// The three behavioural differences a profile encodes are exactly the ones the provider enforces.
/// <see cref="UnsupportedParameters"/> lists the settings the model rejects outright — for the OpenAI families
/// <c>temperature</c> and <c>top_p</c>. <see cref="UsesMaxCompletionTokens"/> says the answer-length cap must be
/// sent as <c>max_completion_tokens</c> rather than <c>max_tokens</c>. <see cref="SupportsReasoningEffort"/> says a
/// deliberation level may be attached. <see cref="SystemMessageRole"/> exists for completeness and defaults to
/// <c>system</c>, because upstream applies no special handling to the system message for these models — it builds
/// the same <c>{"role": "system"}</c> entry for every model and branches only on the sampling parameters.
/// </para>
/// <para>
/// A profile is immutable and safe to share between clients and threads. Build custom ones for a provider whose
/// naming differs from OpenAI's, or to add a family the built-in registry does not yet know about, rather than
/// patching a connector.
/// </para>
/// <para><b>For Beginners:</b> Newer "reasoning" models follow slightly different rules from ordinary chat models:
/// they refuse a couple of settings and spell one of them differently. A profile is a small rulebook for one family
/// of such models — "any model whose name starts with <c>o3</c> behaves like this" — so the library can adjust your
/// request automatically instead of letting the provider reject it. You rarely create one yourself; the built-in
/// registry already covers the OpenAI families.</para>
/// </remarks>
public sealed class ReasoningModelProfile
{
    private readonly string[] _prefixes;
    private readonly ReasoningRequestParameter[] _unsupported;

    /// <summary>The wire value used when no system-message rewriting is requested.</summary>
    public const string DefaultSystemMessageRole = "system";

    /// <summary>Initializes a reasoning-model profile.</summary>
    /// <param name="name">A short family name used in diagnostics, for example <c>openai-o-series</c>.</param>
    /// <param name="modelIdPrefixes">
    /// The model-id prefixes this profile claims, matched case-insensitively against the start of the model id.
    /// Must contain at least one non-empty entry.
    /// </param>
    /// <param name="usesMaxCompletionTokens">
    /// Whether the answer-length cap must be sent as <c>max_completion_tokens</c> instead of <c>max_tokens</c>.
    /// </param>
    /// <param name="unsupportedParameters">
    /// The settings the model rejects; <c>null</c> means <see cref="ReasoningRequestParameter.Temperature"/> and
    /// <see cref="ReasoningRequestParameter.TopP"/>, which is what the OpenAI reasoning families reject.
    /// </param>
    /// <param name="supportsReasoningEffort">Whether a <c>reasoning_effort</c> field may be attached.</param>
    /// <param name="defaultReasoningEffort">
    /// The effort used when the caller leaves <see cref="ReasoningEffortLevel.Unspecified"/> configured;
    /// <see cref="ReasoningEffortLevel.Unspecified"/> means the field is omitted entirely.
    /// </param>
    /// <param name="systemMessageRole">
    /// The role name system messages are sent under; <c>null</c> or empty means <see cref="DefaultSystemMessageRole"/>.
    /// </param>
    /// <exception cref="ArgumentNullException"><paramref name="name"/> or <paramref name="modelIdPrefixes"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">
    /// <paramref name="name"/> is empty or white space, <paramref name="modelIdPrefixes"/> has no usable entry, or
    /// <paramref name="unsupportedParameters"/> contains an undefined value.
    /// </exception>
    public ReasoningModelProfile(
        string name,
        IReadOnlyList<string> modelIdPrefixes,
        bool usesMaxCompletionTokens = true,
        IReadOnlyList<ReasoningRequestParameter>? unsupportedParameters = null,
        bool supportsReasoningEffort = true,
        ReasoningEffortLevel defaultReasoningEffort = ReasoningEffortLevel.Unspecified,
        string? systemMessageRole = null)
    {
        Guard.NotNullOrWhiteSpace(name);
        Guard.NotNull(modelIdPrefixes);

        var prefixes = new List<string>(modelIdPrefixes.Count);
        foreach (string prefix in modelIdPrefixes)
        {
            if (string.IsNullOrWhiteSpace(prefix)) continue;
            prefixes.Add(prefix.Trim());
        }

        if (prefixes.Count == 0)
        {
            throw new ArgumentException(
                "A reasoning-model profile needs at least one non-empty model-id prefix.", nameof(modelIdPrefixes));
        }

        if (!Enum.IsDefined(typeof(ReasoningEffortLevel), defaultReasoningEffort))
        {
            throw new ArgumentOutOfRangeException(
                nameof(defaultReasoningEffort), defaultReasoningEffort, "Value must be a defined effort level.");
        }

        ReasoningRequestParameter[] unsupported;
        if (unsupportedParameters is null)
        {
            unsupported = new[] { ReasoningRequestParameter.Temperature, ReasoningRequestParameter.TopP };
        }
        else
        {
            var distinct = new List<ReasoningRequestParameter>(unsupportedParameters.Count);
            foreach (ReasoningRequestParameter parameter in unsupportedParameters)
            {
                if (!Enum.IsDefined(typeof(ReasoningRequestParameter), parameter))
                {
                    throw new ArgumentException(
                        "An unsupported-parameter list cannot contain an undefined value.", nameof(unsupportedParameters));
                }

                if (!distinct.Contains(parameter)) distinct.Add(parameter);
            }

            unsupported = distinct.ToArray();
        }

        _prefixes = prefixes.ToArray();
        _unsupported = unsupported;
        Name = name.Trim();
        UsesMaxCompletionTokens = usesMaxCompletionTokens;
        SupportsReasoningEffort = supportsReasoningEffort;
        DefaultReasoningEffort = defaultReasoningEffort;
        // Narrowed by an explicit null check rather than string.IsNullOrWhiteSpace, which carries no nullable
        // annotation on .NET Framework and would leave the following Trim() flagged there.
        string role = systemMessageRole is null ? string.Empty : systemMessageRole.Trim();
        SystemMessageRole = role.Length == 0 ? DefaultSystemMessageRole : role;
    }

    /// <summary>Gets the short family name used in diagnostics.</summary>
    public string Name { get; }

    /// <summary>Gets the model-id prefixes this profile claims.</summary>
    public IReadOnlyList<string> ModelIdPrefixes => new ReadOnlyCollection<string>(_prefixes);

    /// <summary>Gets whether the answer-length cap is sent as <c>max_completion_tokens</c>.</summary>
    public bool UsesMaxCompletionTokens { get; }

    /// <summary>Gets the settings this model family rejects.</summary>
    public IReadOnlyList<ReasoningRequestParameter> UnsupportedParameters =>
        new ReadOnlyCollection<ReasoningRequestParameter>(_unsupported);

    /// <summary>Gets whether a <c>reasoning_effort</c> field may be attached to the request.</summary>
    public bool SupportsReasoningEffort { get; }

    /// <summary>Gets the effort used when the caller configured none; <c>Unspecified</c> omits the field.</summary>
    public ReasoningEffortLevel DefaultReasoningEffort { get; }

    /// <summary>Gets the role name system messages are sent under, <c>system</c> unless overridden.</summary>
    public string SystemMessageRole { get; }

    /// <summary>Reports whether this profile claims a model id.</summary>
    /// <param name="modelId">The model id to test; <c>null</c> or empty never matches.</param>
    /// <returns><c>true</c> when the id starts with one of <see cref="ModelIdPrefixes"/>, ignoring case.</returns>
    public bool Matches(string? modelId)
    {
        if (modelId is null) return false;
        string trimmed = modelId.Trim();
        if (trimmed.Length == 0) return false;

        foreach (string prefix in _prefixes)
        {
            if (trimmed.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)) return true;
        }

        return false;
    }

    /// <summary>Reports whether this model family rejects a setting.</summary>
    /// <param name="parameter">The setting to test.</param>
    /// <returns><c>true</c> when the setting must be removed from the request.</returns>
    public bool Rejects(ReasoningRequestParameter parameter)
    {
        foreach (ReasoningRequestParameter unsupported in _unsupported)
        {
            if (unsupported == parameter) return true;
        }

        return false;
    }

    /// <summary>Returns the family name and the number of prefixes it claims.</summary>
    /// <returns>A short description with no endpoint or credential in it.</returns>
    public override string ToString() =>
        "ReasoningModelProfile(" + Name + ", " +
        _prefixes.Length.ToString(System.Globalization.CultureInfo.InvariantCulture) + " prefixes)";
}
