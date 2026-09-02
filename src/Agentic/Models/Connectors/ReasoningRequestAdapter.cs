using System.Collections.ObjectModel;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>
/// Shared reasoning-model policy: resolves the profile for a model, applies the configured handling to a rejected
/// setting, and keeps a bounded record of every adjustment so no drop is silent.
/// </summary>
/// <remarks>
/// Both the provider-neutral decorator and the OpenAI connector need identical decisions about which settings are
/// rejected, whether a rejection throws, which effort level applies, and where notices go. Keeping that policy in
/// one internal collaborator means the two entry points cannot drift apart. Every member is safe to call from
/// several threads at once.
/// </remarks>
internal sealed class ReasoningRequestAdapter
{
    private readonly object _gate = new();
    private readonly Dictionary<string, ReasoningModelDiagnostic> _distinct = new(StringComparer.Ordinal);
    private readonly Dictionary<string, long> _counts = new(StringComparer.Ordinal);
    private readonly ReasoningModelProfileRegistry _registry;
    private readonly IReasoningModelDiagnosticSink? _sink;

    internal ReasoningRequestAdapter(ReasoningModelOptions? options)
    {
        ReasoningModelOptions copy = (options ?? new ReasoningModelOptions()).Clone();
        copy.Validate();
        Options = copy;
        _registry = copy.Profiles ?? ReasoningModelProfileRegistry.Default;
        _sink = copy.DiagnosticSink;
    }

    /// <summary>Gets the validated copy of the options this adapter was built from.</summary>
    internal ReasoningModelOptions Options { get; }

    /// <summary>Gets the registry used to recognise reasoning models.</summary>
    internal ReasoningModelProfileRegistry Registry => _registry;

    /// <summary>Resolves the profile for a model, or <c>null</c> when adaptation is off or the model is ordinary.</summary>
    internal ReasoningModelProfile? Find(string? modelId) =>
        Options.Enabled ? _registry.Find(modelId) : null;

    /// <summary>Resolves the effort level to send: the configured one, else the profile's default.</summary>
    internal ReasoningEffortLevel ResolveEffort(ReasoningModelProfile profile) =>
        Options.ReasoningEffort != ReasoningEffortLevel.Unspecified
            ? Options.ReasoningEffort
            : profile.DefaultReasoningEffort;

    /// <summary>
    /// Applies the configured handling to a setting the model rejects: either throws, or records the drop and
    /// returns so the caller can remove the field.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// The configured handling is <see cref="UnsupportedChatParameterHandling.Throw"/>.
    /// </exception>
    internal void RejectOrRecord(string modelId, ReasoningModelProfile profile, ReasoningRequestParameter parameter)
    {
        if (Options.UnsupportedParameterHandling == UnsupportedChatParameterHandling.Throw)
        {
            throw new InvalidOperationException(
                "Model '" + modelId + "' belongs to the reasoning family '" + profile.Name + "', which rejects the " +
                parameter + " setting. Remove it from the chat options, or set " +
                nameof(ReasoningModelOptions.UnsupportedParameterHandling) + " to " +
                nameof(UnsupportedChatParameterHandling.Drop) + " to send the request without it.");
        }

        Record(modelId, profile, parameter, ReasoningParameterAdjustment.Dropped,
            "The " + parameter + " setting was removed because this reasoning model rejects it.");
    }

    /// <summary>Records one adjustment, forwarding it to the configured sink and to the in-memory summary.</summary>
    internal void Record(
        string modelId,
        ReasoningModelProfile profile,
        ReasoningRequestParameter parameter,
        ReasoningParameterAdjustment adjustment,
        string message)
    {
        var diagnostic = new ReasoningModelDiagnostic(modelId, profile.Name, parameter, adjustment, message);

        lock (_gate)
        {
            string key = diagnostic.Key;
            if (_counts.TryGetValue(key, out long count))
            {
                _counts[key] = count + 1L;
            }
            else
            {
                _counts[key] = 1L;
                _distinct[key] = diagnostic;
            }
        }

        if (_sink is null) return;

        // A reporting failure must never turn into a failed model call: the request itself is already correct.
#pragma warning disable CA1031
        try
        {
            _sink.Report(diagnostic);
        }
        catch (Exception)
        {
            // Intentionally swallowed. The adjustment is still retained in the in-memory summary below, so the
            // information is not lost even when the host's sink is broken.
        }
#pragma warning restore CA1031
    }

    /// <summary>Gets one representative record per distinct adjustment, with its occurrence count.</summary>
    internal IReadOnlyDictionary<string, ReasoningModelDiagnosticSummary> GetSummary()
    {
        lock (_gate)
        {
            var summary = new Dictionary<string, ReasoningModelDiagnosticSummary>(_distinct.Count, StringComparer.Ordinal);
            foreach (KeyValuePair<string, ReasoningModelDiagnostic> pair in _distinct)
            {
                summary[pair.Key] = new ReasoningModelDiagnosticSummary(pair.Value, _counts[pair.Key]);
            }

            return new ReadOnlyDictionary<string, ReasoningModelDiagnosticSummary>(summary);
        }
    }
}
