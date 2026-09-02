using System.Net.Http;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using Newtonsoft.Json.Linq;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage, which is in scope project-wide
// via a global using in AiModelBuilder.cs. The agentic subsystem uses the Models type.
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>
/// An OpenAI Chat Completions connector that adapts its request body when the model is a reasoning model: it
/// removes the sampling settings those models reject, sends the answer cap as <c>max_completion_tokens</c>, and
/// attaches a <c>reasoning_effort</c> level when one is configured.
/// </summary>
/// <typeparam name="T">The numeric type used across the AiDotNet ecosystem.</typeparam>
/// <remarks>
/// <para>
/// The OpenAI o-series and GPT-5 families and the GPT-OSS releases enforce three differences from an ordinary chat
/// model, and every one of them is a hard error rather than a warning. They reject <c>temperature</c> and
/// <c>top_p</c>; they refuse <c>max_tokens</c> and require the cap under the name <c>max_completion_tokens</c>; and
/// they accept an optional <c>reasoning_effort</c>. A request built for <c>gpt-4o</c> and pointed at <c>o3</c>
/// therefore fails with an HTTP 400 before a single token is generated. This connector is
/// <see cref="OpenAIChatClient{T}"/> with those three rules applied, so switching a run from <c>gpt-4o</c> to
/// <c>o3</c> is a change of model id and nothing else.
/// </para>
/// <para>
/// Detection is by model-id prefix through a <see cref="ReasoningModelProfileRegistry"/>, never by endpoint. The
/// same weights are reachable through OpenAI, Azure, OpenRouter, and local proxies, and all of them enforce the
/// same rules, so a check on the base URL would be wrong on most of those. When the model id matches no profile —
/// <c>gpt-4o</c>, a fine-tune, a third-party model — the request body is exactly what the base connector produces,
/// byte for byte.
/// </para>
/// <para>
/// Two behaviours go beyond the reference implementation. Every adjustment is recorded and readable through
/// <see cref="GetDiagnosticSummary"/>, so a run can prove that its configured temperature never applied instead of
/// leaving that to be inferred; and <c>UnsupportedParameterHandling</c> can be set to fail loudly rather than drop,
/// for a pipeline where losing the setting silently would invalidate the comparison being made. Upstream builds a
/// separate parameter dictionary and records nothing.
/// </para>
/// <para>
/// The system message is left under the <c>system</c> role by default, which is what upstream sends to these
/// models and what the endpoint accepts. Set <c>RewriteSystemMessageRole</c> and give the profile a different role
/// name only for a provider that insists on the newer <c>developer</c> role.
/// </para>
/// <para><b>For Beginners:</b> Use this instead of <see cref="OpenAIChatClient{T}"/> whenever the model you name
/// might be one of OpenAI's reasoning models (anything starting with <c>o1</c>, <c>o3</c>, <c>o4-</c>, or
/// <c>gpt-5</c>). Those models are fussy about request settings, and this class quietly fixes the request for you —
/// and tells you what it fixed. For every other model it behaves identically to the plain connector, so it is safe
/// to use as your default.</para>
/// </remarks>
public class ReasoningOpenAIChatClient<T> : OpenAIChatClient<T>
{
    private const string TemperatureField = "temperature";
    private const string TopPField = "top_p";
    private const string SeedField = "seed";
    private const string StopField = "stop";
    private const string MaxTokensField = "max_tokens";
    private const string MaxCompletionTokensField = "max_completion_tokens";
    private const string ResponseFormatField = "response_format";
    private const string ToolsField = "tools";
    private const string ToolChoiceField = "tool_choice";
    private const string ReasoningEffortField = "reasoning_effort";
    private const string MessagesField = "messages";
    private const string RoleField = "role";

    private readonly ReasoningRequestAdapter _adapter;

    /// <summary>Initializes a reasoning-aware OpenAI chat client.</summary>
    /// <param name="apiKey">The OpenAI API key. It is used for the <c>Authorization</c> header and nothing else.</param>
    /// <param name="modelName">The model id (default <c>gpt-4o</c>), which is what decides whether adaptation applies.</param>
    /// <param name="endpoint">Optional custom endpoint (defaults to the public OpenAI Chat Completions URL).</param>
    /// <param name="httpClient">Optional HTTP client.</param>
    /// <param name="options">Reasoning-model settings; <c>null</c> uses the defaults and the built-in profiles.</param>
    /// <exception cref="ArgumentNullException"><paramref name="apiKey"/> or <paramref name="modelName"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="apiKey"/> or <paramref name="modelName"/> is empty or white space.</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option value is outside its permitted range.</exception>
    public ReasoningOpenAIChatClient(
        string apiKey,
        string modelName = "gpt-4o",
        string? endpoint = null,
        HttpClient? httpClient = null,
        ReasoningModelOptions? options = null)
        : base(apiKey, modelName, endpoint, httpClient)
    {
        _adapter = new ReasoningRequestAdapter(options);
    }

    /// <summary>Gets the profile matching this client's model, or <c>null</c> when it is an ordinary chat model.</summary>
    public ReasoningModelProfile? Profile => _adapter.Find(ModelId);

    /// <summary>Gets whether this client's model is recognised as a reasoning model.</summary>
    public bool IsReasoningModel => Profile is not null;

    /// <summary>Gets a copy of the reasoning settings in force.</summary>
    /// <returns>An independent copy; mutating it does not affect this client.</returns>
    public ReasoningModelOptions GetOptions() => _adapter.Options.Clone();

    /// <summary>Gets one entry per distinct request adjustment made so far, with how often it occurred.</summary>
    /// <returns>
    /// A dictionary keyed by <see cref="ReasoningModelDiagnostic.Key"/>; empty when nothing has been adjusted.
    /// Bounded by the number of distinct settings, so it stays small on a run of any length.
    /// </returns>
    public IReadOnlyDictionary<string, ReasoningModelDiagnosticSummary> GetDiagnosticSummary() =>
        _adapter.GetSummary();

    /// <inheritdoc/>
    /// <exception cref="InvalidOperationException">
    /// The request carries a setting this reasoning model rejects and the options say to throw rather than drop.
    /// </exception>
    protected override JObject BuildRequest(IReadOnlyList<ChatMessage> messages, ChatOptions options, bool stream)
    {
        JObject payload = base.BuildRequest(messages, options, stream);

        string modelId = ModelId;
        ReasoningModelProfile? profile = _adapter.Find(modelId);
        if (profile is null) return payload;

        DropRejected(payload, modelId, profile);
        SubstituteCompletionTokenCap(payload, modelId, profile);
        AttachReasoningEffort(payload, modelId, profile);
        RewriteSystemRole(payload, modelId, profile);
        return payload;
    }

    private void DropRejected(JObject payload, string modelId, ReasoningModelProfile profile)
    {
        // The cap is handled by SubstituteCompletionTokenCap: a profile that both rejects it and renames it would
        // otherwise remove the field before the rename could see it.
        RemoveIfRejected(payload, modelId, profile, ReasoningRequestParameter.Temperature, TemperatureField);
        RemoveIfRejected(payload, modelId, profile, ReasoningRequestParameter.TopP, TopPField);
        RemoveIfRejected(payload, modelId, profile, ReasoningRequestParameter.Seed, SeedField);
        RemoveIfRejected(payload, modelId, profile, ReasoningRequestParameter.StopSequences, StopField);
        RemoveIfRejected(payload, modelId, profile, ReasoningRequestParameter.ResponseFormat, ResponseFormatField);

        if (profile.Rejects(ReasoningRequestParameter.Tools) && payload[ToolsField] is not null)
        {
            _adapter.RejectOrRecord(modelId, profile, ReasoningRequestParameter.Tools);
            payload.Remove(ToolsField);
            payload.Remove(ToolChoiceField);
        }

        if (profile.Rejects(ReasoningRequestParameter.MaxOutputTokens) && !profile.UsesMaxCompletionTokens)
        {
            RemoveIfRejected(payload, modelId, profile, ReasoningRequestParameter.MaxOutputTokens, MaxTokensField);
        }
    }

    private void RemoveIfRejected(
        JObject payload,
        string modelId,
        ReasoningModelProfile profile,
        ReasoningRequestParameter parameter,
        string field)
    {
        if (!profile.Rejects(parameter) || payload[field] is null) return;
        _adapter.RejectOrRecord(modelId, profile, parameter);
        payload.Remove(field);
    }

    private void SubstituteCompletionTokenCap(JObject payload, string modelId, ReasoningModelProfile profile)
    {
        if (!profile.UsesMaxCompletionTokens) return;
        if (payload[MaxTokensField] is not JToken cap) return;

        payload.Remove(MaxTokensField);
        payload[MaxCompletionTokensField] = cap;
        _adapter.Record(
            modelId,
            profile,
            ReasoningRequestParameter.MaxOutputTokens,
            ReasoningParameterAdjustment.Substituted,
            "The answer cap was sent as max_completion_tokens because this reasoning model rejects max_tokens.");
    }

    private void AttachReasoningEffort(JObject payload, string modelId, ReasoningModelProfile profile)
    {
        if (!profile.SupportsReasoningEffort) return;

        ReasoningEffortLevel effort = _adapter.ResolveEffort(profile);
        string? wireValue = effort.ToWireValue();
        if (wireValue is null) return;

        payload[ReasoningEffortField] = wireValue;
        _adapter.Record(
            modelId,
            profile,
            ReasoningRequestParameter.ReasoningEffort,
            ReasoningParameterAdjustment.Added,
            "A reasoning_effort of '" + wireValue + "' was attached in place of the sampling controls.");
    }

    private void RewriteSystemRole(JObject payload, string modelId, ReasoningModelProfile profile)
    {
        if (!_adapter.Options.RewriteSystemMessageRole) return;
        if (string.Equals(profile.SystemMessageRole, ReasoningModelProfile.DefaultSystemMessageRole, StringComparison.Ordinal))
        {
            return;
        }

        if (payload[MessagesField] is not JArray messages) return;

        bool rewrote = false;
        foreach (JToken message in messages)
        {
            if (message is not JObject entry) continue;
            if ((string?)entry[RoleField] != ReasoningModelProfile.DefaultSystemMessageRole) continue;
            entry[RoleField] = profile.SystemMessageRole;
            rewrote = true;
        }

        if (!rewrote) return;

        _adapter.Record(
            modelId,
            profile,
            ReasoningRequestParameter.SystemMessageRole,
            ReasoningParameterAdjustment.Substituted,
            "System messages were sent under the '" + profile.SystemMessageRole + "' role.");
    }
}
