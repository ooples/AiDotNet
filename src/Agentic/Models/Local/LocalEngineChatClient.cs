using System.Linq;
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
/// decodes incrementally and yields the new text on each step. Constrained decoding <em>is</em> supported via
/// <see cref="LocalEngineOptions.Constraint"/> (an <see cref="ITokenConstraint"/> enforced at the logits, e.g.
/// <see cref="AllowedTokenSetConstraint"/> / <see cref="FiniteStateTokenConstraint"/>). What this slice does
/// not do: native tool-calling and auto-deriving a constraint from <see cref="ChatOptions.ResponseFormat"/> —
/// requests that ask for either are rejected with <see cref="NotSupportedException"/> rather than silently
/// returning plain text; set <see cref="LocalEngineOptions.Constraint"/> explicitly for guaranteed-structured
/// output.
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
    /// <exception cref="NotSupportedException">
    /// Thrown when <paramref name="options"/> requests tools or a non-text response format, which the local
    /// engine does not implement.
    /// </exception>
    public Task<ChatResponse> GetResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ValidateSupportedOptions(options);
        var promptIds = BuildPromptIds(messages);
        var maxTokens = ResolveMaxTokens(options);
        var stopSequences = ResolveStopSequences(options);

        List<int> generated;
        ChatFinishReason finishReason;
        var beamWidth = _options.BeamWidth is { } width && width > 1 ? width : 1;
        if (beamWidth > 1)
        {
            (generated, finishReason) = RunBeamSearch(promptIds, maxTokens, beamWidth, stopSequences, cancellationToken);
        }
        else
        {
            var sampling = ResolveSampling(options);
            var sampler = new TokenSampler<T>(sampling.Seed);
            generated = new List<int>();
            finishReason = _model is IIncrementalCausalLanguageModel<T> incremental
                ? RunGenerationIncremental(incremental, promptIds, maxTokens, sampler, sampling, stopSequences, generated, cancellationToken)
                : RunGeneration(promptIds, maxTokens, sampler, sampling, stopSequences, generated, cancellationToken);
        }

        var text = generated.Count > 0 ? _tokenizer.Decode(generated) : string.Empty;
        text = TrimAtStopSequence(text, stopSequences);
        var usage = new ChatUsage(promptIds.Count, generated.Count);
        var response = new ChatResponse(ChatMessage.Assistant(text), finishReason, usage, ModelId);
        return Task.FromResult(response);
    }

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException">
    /// Thrown when <paramref name="options"/> requests tools or a non-text response format, which the local
    /// engine does not implement.
    /// </exception>
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ValidateSupportedOptions(options);
        await Task.CompletedTask.ConfigureAwait(false);

        var promptIds = BuildPromptIds(messages);
        var sampling = ResolveSampling(options);
        var sampler = new TokenSampler<T>(sampling.Seed);
        var maxTokens = ResolveMaxTokens(options);

        yield return new ChatResponseUpdate(role: ChatRole.Assistant);

        var stopSequences = ResolveStopSequences(options);
        var generated = new List<int>();
        var finishReason = ChatFinishReason.Length;

        // Reuse the incremental (KV-cached) path when the model supports it —
        // re-feeding the full prefix per token would make streaming quadratic
        // in the reply length while non-streaming stays O(1) per token.
        var incremental = _model as IIncrementalCausalLanguageModel<T>;
        List<int>? fullContext = null;
        Vector<T>? logits = null;
        if (incremental is not null)
        {
            incremental.ResetCache();
            logits = incremental.StartSequence(promptIds);
        }
        else
        {
            fullContext = new List<int>(promptIds);
        }

        // Incremental decode (HF TextStreamer pattern): only the tokens since
        // the last newline boundary are re-decoded each step, instead of the
        // entire generated buffer.
        var cacheTokens = new List<int>();
        var committedText = string.Empty;
        var emittedLength = 0;

        for (var step = 0; step < maxTokens; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            int? next;
            if (incremental is not null && logits is { } currentLogits)
            {
                next = PickToken(generated, currentLogits, sampler, sampling);
            }
            else if (fullContext is { } context)
            {
                next = NextToken(context, generated, sampler, sampling);
            }
            else
            {
                next = null;
            }

            if (next is null || next.Value == _tokenizer.EosTokenId)
            {
                finishReason = ChatFinishReason.Stop;
                break;
            }

            generated.Add(next.Value);
            fullContext?.Add(next.Value);
            cacheTokens.Add(next.Value);

            var cacheText = _tokenizer.Decode(cacheTokens);
            var fullText = committedText + cacheText;

            var trimmed = TrimAtStopSequence(fullText, stopSequences);
            var stopHit = trimmed.Length < fullText.Length;
            var emitTarget = stopHit ? trimmed : fullText;
            if (emitTarget.Length > emittedLength)
            {
                yield return ChatResponseUpdate.ForText(emitTarget.Substring(emittedLength));
                emittedLength = emitTarget.Length;
            }

            if (stopHit)
            {
                finishReason = ChatFinishReason.Stop;
                break;
            }

            // Commit at newline boundaries so the re-decoded window stays small.
            if (cacheText.EndsWith("\n", StringComparison.Ordinal))
            {
                committedText = fullText;
                cacheTokens.Clear();
            }

            if (incremental is not null)
            {
                logits = incremental.AppendToken(next.Value);
            }
        }

        yield return ChatResponseUpdate.ForFinish(finishReason, new ChatUsage(promptIds.Count, generated.Count));
    }

    // Fail fast on capabilities the local engine does not implement — silently
    // ignoring a tools/structured-output request would hand the caller plain
    // text with no signal that their options were dropped.
    private static void ValidateSupportedOptions(ChatOptions? options)
    {
        if (options is null)
        {
            return;
        }

        if (options.Tools is { Count: > 0 } && options.ToolChoice != ToolChoiceMode.None)
        {
            throw new NotSupportedException(
                "The local engine does not implement native tool calling. Remove the tools from ChatOptions " +
                "(or set ToolChoice to None) when targeting LocalEngineChatClient.");
        }

        if (options.ResponseFormat is { } format && format != ChatResponseFormatKind.Text)
        {
            throw new NotSupportedException(
                $"The local engine does not implement the '{format}' response format. Use " +
                "LocalEngineOptions.Constraint (e.g. a FiniteStateTokenConstraint) for guaranteed-structured output.");
        }
    }

    private ChatFinishReason RunGeneration(
        IReadOnlyList<int> promptIds,
        int maxTokens,
        TokenSampler<T> sampler,
        LocalSamplingOptions sampling,
        IReadOnlyList<string>? stopSequences,
        List<int> generated,
        CancellationToken cancellationToken)
    {
        var context = new List<int>(promptIds);
        for (var step = 0; step < maxTokens; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var next = NextToken(context, generated, sampler, sampling);
            if (next is null || next.Value == _tokenizer.EosTokenId)
            {
                // null = the constraint reached a terminal state; either way generation is complete.
                return ChatFinishReason.Stop;
            }

            generated.Add(next.Value);
            context.Add(next.Value);

            if (stopSequences is not null && ContainsStopSequence(_tokenizer.Decode(generated), stopSequences))
            {
                return ChatFinishReason.Stop;
            }
        }

        return ChatFinishReason.Length;
    }

    // Incremental (KV-cached) generation: prime with the prompt, then advance one token at a time.
    private ChatFinishReason RunGenerationIncremental(
        IIncrementalCausalLanguageModel<T> model,
        IReadOnlyList<int> promptIds,
        int maxTokens,
        TokenSampler<T> sampler,
        LocalSamplingOptions sampling,
        IReadOnlyList<string>? stopSequences,
        List<int> generated,
        CancellationToken cancellationToken)
    {
        model.ResetCache();
        var logits = model.StartSequence(promptIds);

        for (var step = 0; step < maxTokens; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var next = PickToken(generated, logits, sampler, sampling);
            if (next is null || next.Value == _tokenizer.EosTokenId)
            {
                return ChatFinishReason.Stop;
            }

            generated.Add(next.Value);

            if (stopSequences is not null && ContainsStopSequence(_tokenizer.Decode(generated), stopSequences))
            {
                return ChatFinishReason.Stop;
            }

            if (step < maxTokens - 1)
            {
                logits = model.AppendToken(next.Value);
            }
        }

        return ChatFinishReason.Length;
    }

    // Computes the next token id from precomputed logits, honoring the configured constraint, or null when
    // the constraint signals a terminal state (no valid continuation exists).
    private int? PickToken(
        List<int> generated,
        Vector<T> logits,
        TokenSampler<T> sampler,
        LocalSamplingOptions sampling)
    {
        IReadOnlyCollection<int>? allowed = null;
        if (_options.Constraint is { } constraint)
        {
            allowed = constraint.AllowedNextTokens(generated);
            if (allowed is not null && allowed.Count == 0)
            {
                return null;
            }
        }

        return sampler.Sample(logits, sampling, allowed);
    }

    // Computes the next token id by re-feeding the full context (fallback when the model has no KV-cache).
    private int? NextToken(
        List<int> context,
        List<int> generated,
        TokenSampler<T> sampler,
        LocalSamplingOptions sampling) =>
        PickToken(generated, _model.NextTokenLogits(context), sampler, sampling);

    // Beam search: explore `beamWidth` hypotheses in parallel, expanding each by its top tokens (honoring
    // the constraint), and keep the highest-scoring (length-normalized log-probability) beams. Returns the
    // best completion. Deterministic.
    private (List<int> Tokens, ChatFinishReason Finish) RunBeamSearch(
        IReadOnlyList<int> promptIds,
        int maxTokens,
        int beamWidth,
        IReadOnlyList<string>? stopSequences,
        CancellationToken cancellationToken)
    {
        var beams = new List<Beam> { new(new List<int>(), 0.0, finished: false) };

        for (var step = 0; step < maxTokens; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (beams.All(b => b.Finished))
            {
                break;
            }

            var expanded = new List<Beam>();
            foreach (var beam in beams)
            {
                if (beam.Finished)
                {
                    expanded.Add(beam);
                    continue;
                }

                IReadOnlyCollection<int>? allowed = null;
                if (_options.Constraint is { } constraint)
                {
                    allowed = constraint.AllowedNextTokens(beam.Tokens);
                    if (allowed is not null && allowed.Count == 0)
                    {
                        expanded.Add(new Beam(beam.Tokens, beam.LogProbability, finished: true));
                        continue;
                    }
                }

                var context = new List<int>(promptIds.Count + beam.Tokens.Count);
                context.AddRange(promptIds);
                context.AddRange(beam.Tokens);

                var logProbabilities = LogSoftmax(_model.NextTokenLogits(context), allowed);
                foreach (var (tokenId, logProbability) in TopTokens(logProbabilities, beamWidth))
                {
                    var finished = tokenId == _tokenizer.EosTokenId;
                    var tokens = new List<int>(beam.Tokens);
                    if (!finished)
                    {
                        tokens.Add(tokenId);
                    }

                    if (!finished && stopSequences is not null && ContainsStopSequence(_tokenizer.Decode(tokens), stopSequences))
                    {
                        finished = true;
                    }

                    expanded.Add(new Beam(tokens, beam.LogProbability + logProbability, finished));
                }
            }

            beams = expanded
                .OrderByDescending(NormalizedScore)
                .Take(beamWidth)
                .ToList();
        }

        var best = beams.OrderByDescending(NormalizedScore).First();
        return (best.Tokens, best.Finished ? ChatFinishReason.Stop : ChatFinishReason.Length);
    }

    private static double NormalizedScore(Beam beam) => beam.LogProbability / Math.Max(1, beam.Tokens.Count);

    private static double[] LogSoftmax(Vector<T> logits, IReadOnlyCollection<int>? allowed)
    {
        var count = logits.Length;
        bool[]? allowedMask = null;
        if (allowed is not null)
        {
            allowedMask = new bool[count];
            foreach (var id in allowed)
            {
                if (id >= 0 && id < count)
                {
                    allowedMask[id] = true;
                }
            }
        }

        var scores = new double[count];
        var max = double.NegativeInfinity;
        for (var i = 0; i < count; i++)
        {
            var value = allowedMask is null || allowedMask[i] ? Convert.ToDouble(logits[i]) : double.NegativeInfinity;
            scores[i] = value;
            if (value > max)
            {
                max = value;
            }
        }

        var sumExp = 0.0;
        for (var i = 0; i < count; i++)
        {
            if (!double.IsNegativeInfinity(scores[i]))
            {
                sumExp += Math.Exp(scores[i] - max);
            }
        }

        var logSumExp = max + Math.Log(sumExp);
        for (var i = 0; i < count; i++)
        {
            scores[i] = double.IsNegativeInfinity(scores[i]) ? double.NegativeInfinity : scores[i] - logSumExp;
        }

        return scores;
    }

    private static IEnumerable<(int TokenId, double LogProbability)> TopTokens(double[] logProbabilities, int k)
    {
        return Enumerable.Range(0, logProbabilities.Length)
            .Where(i => !double.IsNegativeInfinity(logProbabilities[i]))
            .OrderByDescending(i => logProbabilities[i])
            .Take(k)
            .Select(i => (i, logProbabilities[i]));
    }

    private sealed class Beam
    {
        public Beam(List<int> tokens, double logProbability, bool finished)
        {
            Tokens = tokens;
            LogProbability = logProbability;
            Finished = finished;
        }

        public List<int> Tokens { get; }

        public double LogProbability { get; }

        public bool Finished { get; }
    }

    private static IReadOnlyList<string>? ResolveStopSequences(ChatOptions? options)
    {
        var stops = options?.StopSequences;
        if (stops is null || stops.Count == 0)
        {
            return null;
        }

        var filtered = stops.Where(s => s is not null && s.Length > 0).ToList();
        return filtered.Count > 0 ? filtered : null;
    }

    private static bool ContainsStopSequence(string text, IReadOnlyList<string> stopSequences)
    {
        foreach (var stop in stopSequences)
        {
            if (text.IndexOf(stop, StringComparison.Ordinal) >= 0)
            {
                return true;
            }
        }

        return false;
    }

    private static string TrimAtStopSequence(string text, IReadOnlyList<string>? stopSequences)
    {
        if (stopSequences is null)
        {
            return text;
        }

        var earliest = -1;
        foreach (var stop in stopSequences)
        {
            var index = text.IndexOf(stop, StringComparison.Ordinal);
            if (index >= 0 && (earliest < 0 || index < earliest))
            {
                earliest = index;
            }
        }

        return earliest >= 0 ? text.Substring(0, earliest) : text;
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
