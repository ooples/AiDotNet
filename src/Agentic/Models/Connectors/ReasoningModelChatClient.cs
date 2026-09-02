using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Validation;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage, which is in scope project-wide
// via a global using in AiModelBuilder.cs. The agentic subsystem uses the Models type.
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>
/// Wraps any <see cref="IChatClient{T}"/> and removes the per-request settings a reasoning model rejects, reporting
/// each removal instead of dropping it in silence.
/// </summary>
/// <typeparam name="T">The numeric type used across the AiDotNet ecosystem, passed through to the wrapped client.</typeparam>
/// <remarks>
/// <para>
/// Reasoning models — the OpenAI o-series and GPT-5 families and the GPT-OSS releases — answer a request carrying
/// <c>temperature</c> or <c>top_p</c> with an HTTP 400. Any pipeline that sets a temperature, which an evolutionary
/// search normally does, therefore fails on its very first call the moment the model id is switched to one of
/// those. This decorator sits in front of whichever connector you already use, resolves the model id against a
/// <see cref="ReasoningModelProfileRegistry"/>, and hands the connector a copy of the options with the rejected
/// settings cleared. The caller's own <see cref="ChatOptions"/> instance is never mutated.
/// </para>
/// <para>
/// It deliberately does one thing. Renaming the answer-length cap to <c>max_completion_tokens</c> and attaching
/// <c>reasoning_effort</c> are wire-format concerns that only the connector can carry out, so for OpenAI-shaped
/// endpoints use <see cref="ReasoningOpenAIChatClient{T}"/>, which does the whole job including the removals this
/// decorator performs. Reach for this decorator when the backend is something else — Azure through a custom
/// connector, a local proxy, a third-party gateway — and you only need the request to stop being rejected.
/// </para>
/// <para>
/// Every removal is recorded. <see cref="GetDiagnosticSummary"/> returns one entry per distinct adjustment with a
/// count, and a <c>DiagnosticSink</c> on the options receives every occurrence. That is the difference from the
/// reference implementation, which builds a different parameter dictionary for reasoning models and leaves no trace
/// that the configured temperature never applied.
/// </para>
/// <para><b>For Beginners:</b> Some newer AI models refuse settings such as "creativity" (temperature). If your
/// code sets one and you point it at such a model, the provider rejects the whole request. Wrap your client in this
/// one — <c>new ReasoningModelChatClient&lt;double&gt;(myClient)</c> — and the offending settings are taken out
/// automatically, with a note telling you which ones. Streaming, tools, and everything else pass straight
/// through.</para>
/// </remarks>
public sealed class ReasoningModelChatClient<T> : IChatClient<T>
{
    private readonly IChatClient<T> _inner;
    private readonly ReasoningRequestAdapter _adapter;

    /// <summary>Initializes a reasoning-aware decorator over an existing chat client.</summary>
    /// <param name="inner">The client that performs the actual call.</param>
    /// <param name="options">Reasoning-model settings; <c>null</c> uses the defaults and the built-in profiles.</param>
    /// <exception cref="ArgumentNullException"><paramref name="inner"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option value is outside its permitted range.</exception>
    public ReasoningModelChatClient(IChatClient<T> inner, ReasoningModelOptions? options = null)
    {
        Guard.NotNull(inner);
        _inner = inner;
        _adapter = new ReasoningRequestAdapter(options);
    }

    /// <inheritdoc/>
    public string ModelId => _inner.ModelId;

    /// <summary>Gets the wrapped client this decorator delegates to.</summary>
    public IChatClient<T> InnerClient => _inner;

    /// <summary>Gets the profile matching the wrapped client's model, or <c>null</c> when it is an ordinary model.</summary>
    public ReasoningModelProfile? Profile => _adapter.Find(_inner.ModelId);

    /// <summary>Gets a copy of the reasoning settings in force.</summary>
    /// <returns>An independent copy; mutating it does not affect this client.</returns>
    public ReasoningModelOptions GetOptions() => _adapter.Options.Clone();

    /// <summary>Gets one entry per distinct adjustment made so far, with how often it occurred.</summary>
    /// <returns>
    /// A dictionary keyed by <see cref="ReasoningModelDiagnostic.Key"/>; empty when nothing has been adjusted.
    /// Bounded by the number of distinct settings, so it stays small on a run of any length.
    /// </returns>
    public IReadOnlyDictionary<string, ReasoningModelDiagnosticSummary> GetDiagnosticSummary() =>
        _adapter.GetSummary();

    /// <summary>Returns the options the wrapped client would be given for a request.</summary>
    /// <param name="options">The caller's options; <c>null</c> is returned unchanged.</param>
    /// <returns>
    /// The same instance when nothing needed changing, or a copy with the rejected settings cleared. The caller's
    /// instance is never mutated.
    /// </returns>
    /// <remarks>
    /// Exposed so a caller can see exactly what a reasoning model will be sent without performing the call. Calling
    /// it also records the adjustments, so use it for inspection rather than in a loop.
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// A rejected setting is present and the options say to throw rather than drop.
    /// </exception>
    public ChatOptions? AdaptOptions(ChatOptions? options)
    {
        if (options is null) return null;

        string modelId = _inner.ModelId;
        ReasoningModelProfile? profile = _adapter.Find(modelId);
        if (profile is null) return options;

        ChatOptions? adapted = null;
        if (options.Temperature is not null && profile.Rejects(ReasoningRequestParameter.Temperature))
        {
            _adapter.RejectOrRecord(modelId, profile, ReasoningRequestParameter.Temperature);
            adapted = Copy(options, adapted);
            adapted.Temperature = null;
        }

        if (options.TopP is not null && profile.Rejects(ReasoningRequestParameter.TopP))
        {
            _adapter.RejectOrRecord(modelId, profile, ReasoningRequestParameter.TopP);
            adapted = Copy(options, adapted);
            adapted.TopP = null;
        }

        if (options.TopK is not null && profile.Rejects(ReasoningRequestParameter.TopK))
        {
            _adapter.RejectOrRecord(modelId, profile, ReasoningRequestParameter.TopK);
            adapted = Copy(options, adapted);
            adapted.TopK = null;
        }

        if (options.Seed is not null && profile.Rejects(ReasoningRequestParameter.Seed))
        {
            _adapter.RejectOrRecord(modelId, profile, ReasoningRequestParameter.Seed);
            adapted = Copy(options, adapted);
            adapted.Seed = null;
        }

        if (options.StopSequences is { Count: > 0 } && profile.Rejects(ReasoningRequestParameter.StopSequences))
        {
            _adapter.RejectOrRecord(modelId, profile, ReasoningRequestParameter.StopSequences);
            adapted = Copy(options, adapted);
            adapted.StopSequences = null;
        }

        if (options.MaxOutputTokens is not null && profile.Rejects(ReasoningRequestParameter.MaxOutputTokens))
        {
            _adapter.RejectOrRecord(modelId, profile, ReasoningRequestParameter.MaxOutputTokens);
            adapted = Copy(options, adapted);
            adapted.MaxOutputTokens = null;
        }

        if (options.ResponseFormat is not null && profile.Rejects(ReasoningRequestParameter.ResponseFormat))
        {
            _adapter.RejectOrRecord(modelId, profile, ReasoningRequestParameter.ResponseFormat);
            adapted = Copy(options, adapted);
            adapted.ResponseFormat = null;
            adapted.ResponseJsonSchema = null;
        }

        if (options.Tools is { Count: > 0 } && profile.Rejects(ReasoningRequestParameter.Tools))
        {
            _adapter.RejectOrRecord(modelId, profile, ReasoningRequestParameter.Tools);
            adapted = Copy(options, adapted);
            adapted.Tools = null;
            adapted.ToolChoice = null;
            adapted.RequiredToolName = null;
        }

        return adapted ?? options;
    }

    /// <inheritdoc/>
    public Task<ChatResponse> GetResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default) =>
        _inner.GetResponseAsync(messages, AdaptOptions(options), cancellationToken);

    /// <inheritdoc/>
    public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default) =>
        _inner.GetStreamingResponseAsync(messages, AdaptOptions(options), cancellationToken);

    private static ChatOptions Copy(ChatOptions source, ChatOptions? existing)
    {
        if (existing is not null) return existing;
        return new ChatOptions
        {
            Temperature = source.Temperature,
            MaxOutputTokens = source.MaxOutputTokens,
            TopP = source.TopP,
            TopK = source.TopK,
            StopSequences = source.StopSequences,
            Seed = source.Seed,
            Tools = source.Tools,
            ToolChoice = source.ToolChoice,
            RequiredToolName = source.RequiredToolName,
            ResponseFormat = source.ResponseFormat,
            ResponseJsonSchema = source.ResponseJsonSchema
        };
    }
}
