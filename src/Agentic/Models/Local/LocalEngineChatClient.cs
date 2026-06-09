using System.Runtime.CompilerServices;
using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// An <see cref="IChatClient{T}"/> that runs entirely in-process over an <see cref="ICausalLanguageModel{T}"/>
/// — no network, no API key, no external service. It renders the conversation to a prompt, encodes it,
/// autoregressively samples tokens until the end-of-sequence token or a length limit, and decodes the
/// result. This is the flagship "local-first" capability: the same agent code that drives OpenAI or
/// Anthropic drives AiDotNet's own model.
/// </summary>
/// <typeparam name="T">The tensor element type shared with the model.</typeparam>
/// <remarks>
/// <para>
/// Because it implements <see cref="IChatClient{T}"/>, the local engine is a drop-in for every higher layer
/// (agents, supervisor/swarm, memory). Both non-streaming and streaming generation are supported; streaming
/// decodes incrementally and yields the new text on each step. Native tool-calling and structured-output
/// constraints are not applied by this slice (they require constrained decoding, a follow-up) — supplied
/// tools are ignored and the engine produces plain text.
/// </para>
/// <para><b>For Beginners:</b> This is your own chatbot brain running on your machine. You hand it the
/// conversation; it writes the reply one word-piece at a time until it decides it's done or hits the length
/// cap. Everything else in this library that talks to a "chat model" can talk to this one instead — so you
/// can build agents with no cloud dependency at all.
/// </para>
/// </remarks>
public sealed class LocalEngineChatClient<T> : IChatClient<T>
{
    private readonly ICausalLanguageModel<T> _model;
    private readonly IGenerationTokenizer _tokenizer;
    private readonly IChatPromptTemplate _template;
    private readonly LocalEngineOptions _options;

    /// <summary>
    /// Initializes a new local engine.
    /// </summary>
    /// <param name="model">The in-process language model that produces next-token logits.</param>
    /// <param name="tokenizer">The tokenizer used to encode prompts and decode generated tokens.</param>
    /// <param name="template">The chat prompt template. <c>null</c> uses <see cref="ChatMlPromptTemplate"/>.</param>
    /// <param name="options">Engine settings. <c>null</c> uses defaults.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="model"/> or <paramref name="tokenizer"/> is <c>null</c>.</exception>
    public LocalEngineChatClient(
        ICausalLanguageModel<T> model,
        IGenerationTokenizer tokenizer,
        IChatPromptTemplate? template = null,
        LocalEngineOptions? options = null)
    {
        Guard.NotNull(model);
        Guard.NotNull(tokenizer);
        _model = model;
        _tokenizer = tokenizer;
        _template = template ?? new ChatMlPromptTemplate();
        _options = options ?? new LocalEngineOptions();
    }

    /// <inheritdoc/>
    public string ModelId =>
        _options.ModelId is { } id && id.Trim().Length > 0 ? id : "local";

    /// <inheritdoc/>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="messages"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="messages"/> is empty.</exception>
    public Task<ChatResponse> GetResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var promptIds = BuildPromptIds(messages);
        var sampling = ResolveSampling(options);
        var sampler = new TokenSampler<T>(sampling.Seed);
        var maxTokens = ResolveMaxTokens(options);

        var generated = new List<int>();
        var finishReason = RunGeneration(promptIds, maxTokens, sampler, sampling, generated, cancellationToken);

        var text = generated.Count > 0 ? _tokenizer.Decode(generated) : string.Empty;
        var usage = new ChatUsage(promptIds.Count, generated.Count);
        var response = new ChatResponse(ChatMessage.Assistant(text), finishReason, usage, ModelId);
        return Task.FromResult(response);
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);

        var promptIds = BuildPromptIds(messages);
        var sampling = ResolveSampling(options);
        var sampler = new TokenSampler<T>(sampling.Seed);
        var maxTokens = ResolveMaxTokens(options);

        yield return new ChatResponseUpdate(role: ChatRole.Assistant);

        var context = new List<int>(promptIds);
        var generated = new List<int>();
        var previousText = string.Empty;
        var finishReason = ChatFinishReason.Length;

        for (var step = 0; step < maxTokens; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var logits = _model.NextTokenLogits(context);
            var next = sampler.Sample(logits, sampling);
            if (next == _tokenizer.EosTokenId)
            {
                finishReason = ChatFinishReason.Stop;
                break;
            }

            generated.Add(next);
            context.Add(next);

            var fullText = _tokenizer.Decode(generated);
            if (fullText.Length > previousText.Length)
            {
                var delta = fullText.Substring(previousText.Length);
                previousText = fullText;
                yield return ChatResponseUpdate.ForText(delta);
            }
        }

        yield return ChatResponseUpdate.ForFinish(finishReason, new ChatUsage(promptIds.Count, generated.Count));
    }

    private ChatFinishReason RunGeneration(
        IReadOnlyList<int> promptIds,
        int maxTokens,
        TokenSampler<T> sampler,
        LocalSamplingOptions sampling,
        List<int> generated,
        CancellationToken cancellationToken)
    {
        var context = new List<int>(promptIds);
        for (var step = 0; step < maxTokens; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var logits = _model.NextTokenLogits(context);
            var next = sampler.Sample(logits, sampling);
            if (next == _tokenizer.EosTokenId)
            {
                return ChatFinishReason.Stop;
            }

            generated.Add(next);
            context.Add(next);
        }

        return ChatFinishReason.Length;
    }

    private List<int> BuildPromptIds(IReadOnlyList<ChatMessage> messages)
    {
        Guard.NotNull(messages);
        if (messages.Count == 0)
        {
            throw new ArgumentException("The conversation must contain at least one message.", nameof(messages));
        }

        var prompt = _template.Render(messages);
        return new List<int>(_tokenizer.Encode(prompt));
    }

    private int ResolveMaxTokens(ChatOptions? options)
    {
        if (options?.MaxOutputTokens is { } requested && requested > 0)
        {
            return requested;
        }

        return _options.MaxOutputTokens is { } configured && configured > 0
            ? configured
            : LocalEngineOptions.DefaultMaxOutputTokens;
    }

    private LocalSamplingOptions ResolveSampling(ChatOptions? options)
    {
        var defaults = _options.Sampling ?? new LocalSamplingOptions();
        return new LocalSamplingOptions
        {
            Temperature = options?.Temperature ?? defaults.Temperature,
            TopK = options?.TopK ?? defaults.TopK,
            TopP = options?.TopP ?? defaults.TopP,
            Seed = options?.Seed ?? defaults.Seed,
        };
    }
}
