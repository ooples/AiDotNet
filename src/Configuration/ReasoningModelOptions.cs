using AiDotNet.Agentic.Models.Connectors;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Configuration;

/// <summary>Configures how requests are adapted before they are sent to a reasoning model.</summary>
/// <remarks>
/// <para>
/// The defaults reproduce the reference OpenEvolve request bodies: the built-in profile registry, no
/// <c>reasoning_effort</c> field unless one is asked for, unsupported sampling settings dropped rather than
/// rejected, and system messages left under the <c>system</c> role. The one deliberate difference is that every
/// dropped setting is reported, so a run can prove what it actually sent.
/// </para>
/// <para>
/// <see cref="ReasoningEffort"/> is the setting most callers change. Leaving it
/// <see cref="ReasoningEffortLevel.Unspecified"/> omits the field entirely and lets the provider choose; raising it
/// buys better answers on hard proposals at a real cost in latency and hidden reasoning tokens, which a search
/// pays on every iteration.
/// </para>
/// <para>
/// <see cref="RewriteSystemMessageRole"/> is off by default and should normally stay off. The reference
/// implementation applies no special handling to the system message for reasoning models — it builds the same
/// <c>system</c> entry for every model — and the OpenAI Chat Completions endpoint still accepts that role for the
/// o-series and GPT-5 families. Turn it on only for a provider that demands the newer <c>developer</c> role, and
/// name that role on the profile.
/// </para>
/// <para><b>For Beginners:</b> Newer "reasoning" models refuse a couple of the settings ordinary chat models
/// accept. These options control how the library copes: which model names count as reasoning models, how hard to
/// ask them to think, whether to quietly drop a rejected setting or stop with an error, and where to send the
/// notices about what it changed. The defaults are safe; the one worth trying is
/// <see cref="ReasoningEffort"/>.</para>
/// </remarks>
public sealed class ReasoningModelOptions
{
    /// <summary>Gets or sets whether reasoning-model adaptation runs at all.</summary>
    /// <remarks>
    /// Set to <c>false</c> to send requests exactly as configured, which is useful when a proxy already normalizes
    /// them. A reasoning model will then reject a request that carries a temperature.
    /// </remarks>
    public bool Enabled { get; set; } = true;

    /// <summary>Gets or sets the deliberation level requested, or <c>Unspecified</c> to send no such field.</summary>
    public ReasoningEffortLevel ReasoningEffort { get; set; } = ReasoningEffortLevel.Unspecified;

    /// <summary>Gets or sets what happens to a setting the target model does not support.</summary>
    public UnsupportedChatParameterHandling UnsupportedParameterHandling { get; set; } =
        UnsupportedChatParameterHandling.Drop;

    /// <summary>Gets or sets whether system messages are re-roled to the profile's role name.</summary>
    /// <remarks>
    /// Off by default, matching the reference implementation, which sends the <c>system</c> role to every model.
    /// </remarks>
    public bool RewriteSystemMessageRole { get; set; }

    /// <summary>Gets or sets the profile registry used to recognise reasoning models; <c>null</c> uses the built-ins.</summary>
    public ReasoningModelProfileRegistry? Profiles { get; set; }

    /// <summary>Gets or sets where adjustment notices are sent; <c>null</c> keeps them in the client only.</summary>
    /// <remarks>
    /// Even with no sink, a client retains a bounded summary of the adjustments it made, so a dropped setting is
    /// never invisible. A sink is how those notices reach the host's own logging.
    /// </remarks>
    public IReasoningModelDiagnosticSink? DiagnosticSink { get; set; }

    /// <summary>Creates an independent copy of these options.</summary>
    /// <returns>A copy; the profile registry and sink are shared because both are immutable or thread-safe by contract.</returns>
    public ReasoningModelOptions Clone() => new()
    {
        Enabled = Enabled,
        ReasoningEffort = ReasoningEffort,
        UnsupportedParameterHandling = UnsupportedParameterHandling,
        RewriteSystemMessageRole = RewriteSystemMessageRole,
        Profiles = Profiles,
        DiagnosticSink = DiagnosticSink
    };

    /// <summary>Validates the option values.</summary>
    /// <exception cref="ArgumentOutOfRangeException">An enumeration value is not defined.</exception>
    public void Validate()
    {
        if (!Enum.IsDefined(typeof(ReasoningEffortLevel), ReasoningEffort))
        {
            throw new ArgumentOutOfRangeException(
                nameof(ReasoningEffort), ReasoningEffort, "Value must be a defined effort level.");
        }

        if (!Enum.IsDefined(typeof(UnsupportedChatParameterHandling), UnsupportedParameterHandling))
        {
            throw new ArgumentOutOfRangeException(
                nameof(UnsupportedParameterHandling),
                UnsupportedParameterHandling,
                "Value must be a defined handling mode.");
        }
    }
}
