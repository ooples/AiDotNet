using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Models.OpenAi;
using AiDotNet.Serving.Observability;
using AiDotNet.Serving.Services;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Validation;
using Microsoft.AspNetCore.Mvc;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Serving.Controllers;

/// <summary>
/// OpenAI-compatible API surface (<c>/v1/chat/completions</c>, <c>/v1/completions</c>, <c>/v1/models</c>)
/// so AiDotNet can be driven by any OpenAI client and benchmarked apples-to-apples against vLLM / TGI.
/// Bridges the text-based OpenAI protocol to the token-ID-native generation engine via a per-model tokenizer.
/// </summary>
[ApiController]
[Produces("application/json")]
public sealed class OpenAiController : ControllerBase
{
    private const int DefaultMaxTokens = 128;

    private readonly IModelRepository _modelRepository;
    private readonly ITokenizerRegistry _tokenizers;
    private readonly ITextGenerationService _textGeneration;
    private readonly ILogger<OpenAiController> _logger;
    private readonly Configuration.ServingLimitsOptions _limits;

    public OpenAiController(
        IModelRepository modelRepository,
        ITokenizerRegistry tokenizers,
        ITextGenerationService textGeneration,
        ILogger<OpenAiController> logger,
        Configuration.ServingLimitsOptions? limits = null)
    {
        Guard.NotNull(modelRepository);
        _modelRepository = modelRepository;
        Guard.NotNull(tokenizers);
        _tokenizers = tokenizers;
        Guard.NotNull(textGeneration);
        _textGeneration = textGeneration;
        Guard.NotNull(logger);
        _logger = logger;
        // Optional: when no ServingLimitsOptions is registered in DI, the defaults apply (limits stay active).
        _limits = limits ?? new Configuration.ServingLimitsOptions();
    }

    // Enforces the configured per-request generation limits at the HTTP boundary. Returns an OpenAI 400
    // error result when a limit is exceeded (rejected, not silently clamped), or null when within limits.
    private ObjectResult? CheckLimits(int promptTokens, int requestedMaxTokens, int requestedN)
    {
        if (requestedN < 1 || requestedN > _limits.EffectiveMaxN)
        {
            return OpenAiError(StatusCodes.Status400BadRequest, $"'n' must be between 1 and {_limits.EffectiveMaxN}.");
        }
        if (requestedMaxTokens < 1 || requestedMaxTokens > _limits.EffectiveMaxCompletionTokens)
        {
            return OpenAiError(StatusCodes.Status400BadRequest, $"'max_tokens' must be between 1 and {_limits.EffectiveMaxCompletionTokens}.");
        }
        if (promptTokens + requestedMaxTokens > _limits.EffectiveMaxContextTokens)
        {
            return OpenAiError(StatusCodes.Status400BadRequest,
                $"prompt ({promptTokens} tokens) + max_tokens ({requestedMaxTokens}) exceeds the maximum context length of {_limits.EffectiveMaxContextTokens} tokens.");
        }
        return null;
    }

    // ============================ /v1/models ============================

    /// <summary>Lists loaded models in OpenAI format.</summary>
    [HttpGet("/v1/models")]
    [ProducesResponseType(typeof(ModelList), StatusCodes.Status200OK)]
    public ActionResult<ModelList> ListModels()
    {
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var list = new ModelList
        {
            Data = _modelRepository.GetAllModelInfo()
                .Select(m => new ModelCard { Id = m.Name, Created = created })
                .ToList()
        };
        return Ok(list);
    }

    /// <summary>Returns a single model in OpenAI format.</summary>
    [HttpGet("/v1/models/{modelId}")]
    [ProducesResponseType(typeof(ModelCard), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public ActionResult<ModelCard> GetModel(string modelId)
    {
        var info = _modelRepository.GetModelInfo(modelId);
        if (info == null) return OpenAiError(StatusCodes.Status404NotFound, $"The model '{modelId}' does not exist.", "invalid_request_error", "model_not_found");
        return Ok(new ModelCard { Id = info.Name, Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds() });
    }

    // ============================ /v1/chat/completions ============================

    /// <summary>OpenAI-compatible chat completions (streaming and non-streaming).</summary>
    [HttpPost("/v1/chat/completions")]
    public async Task<IActionResult> ChatCompletions([FromBody] ChatCompletionRequest request, CancellationToken ct)
    {
        if (request == null || request.Messages == null || request.Messages.Count == 0)
            return OpenAiError(StatusCodes.Status400BadRequest, "'messages' is required and cannot be empty.");

        var sw = Stopwatch.StartNew();

        // Multi-LoRA: the model field may be "baseModel@adapter" to serve a shared base with a per-request
        // adapter. Resolve the base model for lookup and carry the adapter through to the engine.
        string baseModel;
        string? adapterId;
        try
        {
            (baseModel, adapterId) = SplitModelAndAdapter(request.Model);
        }
        catch (ArgumentException ex)
        {
            return OpenAiError(StatusCodes.Status400BadRequest, ex.Message, "invalid_request_error", "model");
        }
        if (!TryPrepare(baseModel, out var ctx, out var error))
            return error!;

        string prompt = ChatTemplate.Render(request.Messages.Select(m => (m.Role, m.TextContent())));
        // Builds a FRESH request every call. The structured-output / tool Constraint is a STATEFUL state
        // machine, so each completion (every one of the `n` choices) must get its own instance — a shared
        // constraint would remain in its terminal state after the first choice and corrupt the rest.
        (SpeculativeDecodingRequest Sdr, bool ToolMode) BuildSdr()
        {
            var r = BuildRequest(ctx, prompt, request.ResolveMaxTokens(DefaultMaxTokens),
                request.Temperature, request.TopP, request.TopK, request.MinP, request.ResponseFormat, request.LogitBias,
                request.FrequencyPenalty, request.PresencePenalty, request.Logprobs, request.TopLogprobs);

            // Function calling: constrain the output to a valid tool call whose arguments match the tool's
            // JSON schema (reuses the structured-output engine). Takes precedence over response_format.
            var (toolConstraint, useTools) = ToolConstraintFactory.Build(request.Tools, request.ToolChoice, ctx.Tokenizer, ctx.EosTokenId ?? -1);
            if (useTools)
            {
                r.Constraint = toolConstraint;
            }
            r.AdapterId = adapterId;
            r.Seed = request.Seed;
            return (r, useTools);
        }

        SpeculativeDecodingRequest sdr;
        bool toolMode;
        try
        {
            (sdr, toolMode) = BuildSdr();
        }
        catch (ArgumentException ex)
        {
            ServingMetrics.RecordRequest(success: false, promptTokens: 0, generationTokens: 0, durationSeconds: sw.Elapsed.TotalSeconds);
            return OpenAiError(StatusCodes.Status400BadRequest, ex.Message, "invalid_request_error", "response_format");
        }
        // Enforce configured generation limits at the HTTP boundary (reject, don't silently clamp). ctx's
        // prompt-token count is populated by BuildRequest above.
        var limitError = CheckLimits(ctx.PromptTokenCount, request.ResolveMaxTokens(DefaultMaxTokens), request.N ?? 1);
        if (limitError is not null)
        {
            ServingMetrics.RecordRequest(success: false, promptTokens: ctx.PromptTokenCount, generationTokens: 0, durationSeconds: sw.Elapsed.TotalSeconds);
            return limitError;
        }

        var stops = request.ResolveStop();
        string id = "chatcmpl-" + Guid.NewGuid().ToString("N");
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        // The streaming path implements neither multi-choice (`n`) nor tool-call deltas. Reject those
        // combinations explicitly rather than silently returning a single choice / streaming the raw
        // constrained JSON as assistant content (which would violate the request contract).
        if (request.Stream && (request.N ?? 1) != 1)
        {
            ServingMetrics.RecordRequest(success: false, promptTokens: ctx.PromptTokenCount, generationTokens: 0, durationSeconds: sw.Elapsed.TotalSeconds);
            return OpenAiError(StatusCodes.Status400BadRequest, "Streaming with 'n' greater than 1 is not supported.");
        }
        if (request.Stream && toolMode)
        {
            ServingMetrics.RecordRequest(success: false, promptTokens: ctx.PromptTokenCount, generationTokens: 0, durationSeconds: sw.Elapsed.TotalSeconds);
            return OpenAiError(StatusCodes.Status400BadRequest, "Streaming tool calls are not supported; use stream=false when passing tools.");
        }

        if (!request.Stream)
        {
            // n completions (OpenAI `n`): generate independent completions of the same prompt. At
            // temperature > 0 each draws from its own RNG so they differ; at temperature 0 (greedy) they
            // are deterministically identical, which is the expected behavior. `n` was already validated
            // against the configured limit above (rejected, not clamped).
            int n = request.N ?? 1;
            var response = new ChatCompletionResponse { Id = id, Created = created, Model = request.Model };
            int completionTokens = 0;
            for (int i = 0; i < n; i++)
            {
                // Fresh request + constraint per choice: the constraint is stateful, so choices after the
                // first must not reuse (and resume from) the previous choice's completed constraint.
                var choiceSdr = i == 0 ? sdr : BuildSdr().Sdr;
                string text; int genCount; string finish; ChatLogProbs? logProbs = null;
                try
                {
                    if (request.Logprobs == true || toolMode)
                    {
                        // The batch path surfaces logprobs and reliably drives a constrained generation to
                        // completion; the streaming Collect path is used only for plain free-form output.
                        (text, genCount, finish, logProbs) = await CollectWithLogProbsAsync(ctx, choiceSdr, stops, ct);
                    }
                    else
                    {
                        (text, genCount, finish) = Collect(ctx, choiceSdr, stops, ct);
                    }
                }
                catch (GenerationFailedException gfe)
                {
                    // A generation-engine failure must surface as a 500, not a truncated 200.
                    ServingMetrics.RecordRequest(success: false, promptTokens: ctx.PromptTokenCount, generationTokens: completionTokens, durationSeconds: sw.Elapsed.TotalSeconds);
                    return OpenAiError(StatusCodes.Status500InternalServerError, gfe.Message, "server_error");
                }
                if (toolMode)
                {
                    // Parse the constrained JSON into a tool call. Content is null when the model called a tool.
                    var toolCalls = ToolConstraintFactory.Parse(text, "call_" + Guid.NewGuid().ToString("N"));
                    response.Choices.Add(new ChatChoice
                    {
                        Index = i,
                        Message = new ChatMessageOut { Role = "assistant", Content = null, ToolCalls = toolCalls },
                        FinishReason = toolCalls is not null ? "tool_calls" : finish,
                        LogProbs = logProbs
                    });
                }
                else
                {
                    response.Choices.Add(new ChatChoice { Index = i, Message = new ChatMessageOut { Role = "assistant", Content = text }, FinishReason = finish, LogProbs = logProbs });
                }
                completionTokens += genCount;
            }
            response.Usage = new Usage { PromptTokens = ctx.PromptTokenCount, CompletionTokens = completionTokens, TotalTokens = ctx.PromptTokenCount + completionTokens };
            ServingMetrics.RecordRequest(success: true, promptTokens: ctx.PromptTokenCount, generationTokens: completionTokens, durationSeconds: sw.Elapsed.TotalSeconds);
            return Ok(response);
        }

        // Streaming (SSE).
        BeginSse();
        // First chunk announces the assistant role.
        await WriteChunkAsync(new ChatCompletionChunk
        {
            Id = id, Created = created, Model = request.Model,
            Choices = { new ChatChoice { Index = 0, Delta = new ChatMessageOut { Role = "assistant", Content = string.Empty } } }
        }, ct);

        var ids = new List<int>();
        string prev = string.Empty;
        string lastFull = string.Empty;
        string finishReason = "length";
        bool stopped = false;
        double? ttftSeconds = null;

        foreach (int tok in _textGeneration.GenerateStream(ctx.ModelName, ctx.NumericType, sdr, ct))
        {
            ids.Add(tok);
            ttftSeconds ??= sw.Elapsed.TotalSeconds;
            string full = ctx.Tokenizer.Decode(ids, skipSpecialTokens: true);
            lastFull = full;

            int stopAt = FindStop(full, stops);
            if (stopAt >= 0)
            {
                string delta = stopAt > prev.Length ? full.Substring(prev.Length, stopAt - prev.Length) : string.Empty;
                if (delta.Length > 0)
                    await WriteChunkAsync(ContentChunk(id, created, request.Model, delta), ct);
                prev = full.Substring(0, stopAt);
                stopped = true;
                break;
            }

            // Emit only text that cannot begin a future stop string: hold back the longest suffix of `full`
            // that is a proper prefix of some stop (so "EN" of stop "END" is never streamed before "D"
            // reveals or refutes the match). Streamed content cannot be retracted, hence the buffering.
            int safeLen = full.Length - PendingStopPrefixLen(full, stops);
            if (safeLen > prev.Length)
            {
                string delta = full.Substring(prev.Length, safeLen - prev.Length);
                prev = full.Substring(0, safeLen);
                await WriteChunkAsync(ContentChunk(id, created, request.Model, delta), ct);
            }
        }

        // No stop matched: the held-back tail turned out to be safe — flush it.
        if (!stopped && lastFull.Length > prev.Length)
        {
            await WriteChunkAsync(ContentChunk(id, created, request.Model, lastFull.Substring(prev.Length)), ct);
        }

        finishReason = stopped ? "stop" : (ids.Count >= sdr.MaxNewTokens ? "length" : "stop");
        await WriteChunkAsync(new ChatCompletionChunk
        {
            Id = id, Created = created, Model = request.Model,
            Choices = { new ChatChoice { Index = 0, Delta = new ChatMessageOut(), FinishReason = finishReason } }
        }, ct);
        // Per the OpenAI stream_options.include_usage contract, emit a final chunk (empty choices) carrying the
        // authoritative token usage, so streaming clients get exact counts instead of counting content chunks.
        if (request.StreamOptions?.IncludeUsage == true)
        {
            await WriteChunkAsync(new ChatCompletionChunk
            {
                Id = id, Created = created, Model = request.Model,
                Usage = new Usage
                {
                    PromptTokens = ctx.PromptTokenCount,
                    CompletionTokens = ids.Count,
                    TotalTokens = ctx.PromptTokenCount + ids.Count
                }
            }, ct);
        }
        await WriteDoneAsync(ct);
        RecordStreamMetrics(ctx.PromptTokenCount, ids.Count, sw.Elapsed.TotalSeconds, ttftSeconds);
        return new EmptyResult();
    }

    // ============================ /v1/completions ============================

    /// <summary>OpenAI-compatible text completions (streaming and non-streaming).</summary>
    [HttpPost("/v1/completions")]
    public async Task<IActionResult> Completions([FromBody] CompletionRequest request, CancellationToken ct)
    {
        if (request == null || request.Prompt == null)
            return OpenAiError(StatusCodes.Status400BadRequest, "'prompt' is required.");

        var sw = Stopwatch.StartNew();

        if (!TryPrepare(request.Model, out var ctx, out var error))
            return error!;

        SpeculativeDecodingRequest sdr;
        try
        {
            sdr = BuildRequest(ctx, request.PromptText(), request.ResolveMaxTokens(DefaultMaxTokens),
                request.Temperature, request.TopP, request.TopK, request.MinP, request.ResponseFormat, request.LogitBias,
                request.FrequencyPenalty, request.PresencePenalty);
            sdr.Seed = request.Seed;
        }
        catch (ArgumentException ex)
        {
            ServingMetrics.RecordRequest(success: false, promptTokens: 0, generationTokens: 0, durationSeconds: sw.Elapsed.TotalSeconds);
            return OpenAiError(StatusCodes.Status400BadRequest, ex.Message, "invalid_request_error", "response_format");
        }
        // Enforce configured generation limits at the HTTP boundary (completions has no `n`, so n = 1).
        var limitError = CheckLimits(ctx.PromptTokenCount, request.ResolveMaxTokens(DefaultMaxTokens), 1);
        if (limitError is not null)
        {
            ServingMetrics.RecordRequest(success: false, promptTokens: ctx.PromptTokenCount, generationTokens: 0, durationSeconds: sw.Elapsed.TotalSeconds);
            return limitError;
        }

        var stops = request.ResolveStop();
        string id = "cmpl-" + Guid.NewGuid().ToString("N");
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        if (!request.Stream)
        {
            var (text, genCount, finish) = Collect(ctx, sdr, stops, ct);
            var response = new CompletionResponse
            {
                Id = id, Created = created, Model = request.Model,
                Choices = { new CompletionChoice { Index = 0, Text = text, FinishReason = finish } },
                Usage = new Usage { PromptTokens = ctx.PromptTokenCount, CompletionTokens = genCount, TotalTokens = ctx.PromptTokenCount + genCount }
            };
            ServingMetrics.RecordRequest(success: true, promptTokens: ctx.PromptTokenCount, generationTokens: genCount, durationSeconds: sw.Elapsed.TotalSeconds);
            return Ok(response);
        }

        BeginSse();
        var ids = new List<int>();
        string prev = string.Empty;
        string lastFull = string.Empty;
        bool stopped = false;
        double? ttftSeconds = null;

        foreach (int tok in _textGeneration.GenerateStream(ctx.ModelName, ctx.NumericType, sdr, ct))
        {
            ids.Add(tok);
            ttftSeconds ??= sw.Elapsed.TotalSeconds;
            string full = ctx.Tokenizer.Decode(ids, skipSpecialTokens: true);
            lastFull = full;

            int stopAt = FindStop(full, stops);
            if (stopAt >= 0)
            {
                string delta = stopAt > prev.Length ? full.Substring(prev.Length, stopAt - prev.Length) : string.Empty;
                if (delta.Length > 0)
                    await WriteChunkAsync(CompletionChunk(id, created, request.Model, delta, null), ct);
                prev = full.Substring(0, stopAt);
                stopped = true;
                break;
            }

            // Hold back the longest suffix that could still begin a stop string (see PendingStopPrefixLen);
            // streamed content cannot be retracted, so a partial stop prefix must never be emitted early.
            int safeLen = full.Length - PendingStopPrefixLen(full, stops);
            if (safeLen > prev.Length)
            {
                string delta = full.Substring(prev.Length, safeLen - prev.Length);
                prev = full.Substring(0, safeLen);
                await WriteChunkAsync(CompletionChunk(id, created, request.Model, delta, null), ct);
            }
        }

        if (!stopped && lastFull.Length > prev.Length)
        {
            await WriteChunkAsync(CompletionChunk(id, created, request.Model, lastFull.Substring(prev.Length), null), ct);
        }

        string finishReason = stopped ? "stop" : (ids.Count >= sdr.MaxNewTokens ? "length" : "stop");
        await WriteChunkAsync(CompletionChunk(id, created, request.Model, string.Empty, finishReason), ct);
        // OpenAI stream_options.include_usage: final chunk (empty choices) carries authoritative token usage.
        if (request.StreamOptions?.IncludeUsage == true)
        {
            await WriteChunkAsync(new CompletionResponse
            {
                Id = id, Created = created, Model = request.Model,
                Usage = new Usage
                {
                    PromptTokens = ctx.PromptTokenCount,
                    CompletionTokens = ids.Count,
                    TotalTokens = ctx.PromptTokenCount + ids.Count
                }
            }, ct);
        }
        await WriteDoneAsync(ct);
        RecordStreamMetrics(ctx.PromptTokenCount, ids.Count, sw.Elapsed.TotalSeconds, ttftSeconds);
        return new EmptyResult();
    }

    // ============================ Shared helpers ============================

    /// <summary>Per-request generation context: resolved model, numeric type, tokenizer, prompt tokens.</summary>
    private sealed class GenContext
    {
        public required string ModelName { get; init; }
        public required Configuration.NumericType NumericType { get; init; }
        public required ITokenizer Tokenizer { get; init; }
        public int[] PromptTokenIds { get; set; } = Array.Empty<int>();
        public int PromptTokenCount => PromptTokenIds.Length;
        public int? EosTokenId { get; set; }
    }

    /// <summary>Resolves model, tokenizer, and generation capability, or returns an OpenAI error.</summary>
    private bool TryPrepare(string modelName, out GenContext ctx, out IActionResult? error)
    {
        ctx = null!;
        error = null;

        if (string.IsNullOrWhiteSpace(modelName))
        {
            error = OpenAiError(StatusCodes.Status400BadRequest, "'model' is required.");
            return false;
        }

        var info = _modelRepository.GetModelInfo(modelName);
        if (info == null)
        {
            error = OpenAiError(StatusCodes.Status404NotFound, $"The model '{modelName}' does not exist.", "invalid_request_error", "model_not_found");
            return false;
        }

        if (!_textGeneration.SupportsGeneration(modelName, info.NumericType))
        {
            error = OpenAiError(StatusCodes.Status400BadRequest,
                $"The model '{modelName}' does not support text generation.", "invalid_request_error", "unsupported_model");
            return false;
        }

        if (!_tokenizers.TryGet(modelName, out var tokenizer))
        {
            error = OpenAiError(StatusCodes.Status400BadRequest,
                $"No tokenizer is registered for model '{modelName}'. Load the model with a tokenizer path to use the OpenAI API.",
                "invalid_request_error", "no_tokenizer");
            return false;
        }

        int eos = ResolveEosTokenId(tokenizer);
        ctx = new GenContext
        {
            ModelName = modelName,
            NumericType = info.NumericType,
            Tokenizer = tokenizer,
            EosTokenId = eos >= 0 ? eos : null
        };
        return true;
    }

    /// <summary>Encodes the prompt and builds the engine request. When <paramref name="responseFormat"/> is
    /// supplied it compiles a structured-output constraint (json_object / json_schema / regex); a malformed
    /// response_format throws <see cref="ArgumentException"/>, which the caller maps to a 400.</summary>
    private static SpeculativeDecodingRequest BuildRequest(GenContext ctx, string prompt, int maxTokens, double? temperature, double? topP, int? topK, double? minP, JToken? responseFormat = null, JObject? logitBias = null, double? frequencyPenalty = null, double? presencePenalty = null, bool? logprobs = null, int? topLogprobs = null)
    {
        ctx.PromptTokenIds = ctx.Tokenizer.Encode(prompt).TokenIds.ToArray();
        return new SpeculativeDecodingRequest
        {
            InputTokens = ctx.PromptTokenIds,
            MaxNewTokens = Math.Max(1, maxTokens),
            Temperature = temperature ?? 1.0,
            TopP = topP ?? 1.0,
            TopK = topK ?? 0,
            MinP = minP ?? 0.0,
            EosTokenId = ctx.EosTokenId,
            Constraint = StructuredOutputFactory.Build(responseFormat, ctx.Tokenizer, ctx.EosTokenId ?? -1),
            LogitBias = ParseLogitBias(logitBias),
            FrequencyPenalty = frequencyPenalty ?? 0.0,
            PresencePenalty = presencePenalty ?? 0.0,
            Logprobs = logprobs ?? false,
            TopLogprobs = Math.Clamp(topLogprobs ?? 0, 0, 20),
        };
    }

    /// <summary>Runs the batch (non-streaming) generation path, which surfaces per-token log-probabilities,
    /// applies stop strings to the text, and builds the OpenAI <c>logprobs</c> structure.</summary>
    private async Task<(string Text, int Count, string Finish, ChatLogProbs? LogProbs)> CollectWithLogProbsAsync(
        GenContext ctx, SpeculativeDecodingRequest sdr, IReadOnlyList<string> stops, CancellationToken ct)
    {
        var resp = await _textGeneration.GenerateAsync(ctx.ModelName, ctx.NumericType, sdr, ct).ConfigureAwait(false);

        // Propagate engine/generation failures instead of building a successful (but truncated) 200. A client
        // cancellation surfaces as OperationCanceledException (the framework handles it); any other reported
        // error is a server-side generation failure that must reach the caller as a 500.
        if (resp.Error is { } err)
        {
            ct.ThrowIfCancellationRequested();
            throw new GenerationFailedException(err);
        }

        var genTokens = new List<int>(resp.GeneratedTokens);
        string text = ctx.Tokenizer.Decode(genTokens, skipSpecialTokens: true);
        int stopAt = FindStop(text, stops);
        bool stopped = stopAt >= 0;

        // When a stop string ends generation early, trim the reported token count AND the log-probs to the
        // SAME decoded boundary as the text, so usage/logprobs never include tokens generated after the stop.
        int keptTokens = genTokens.Count;
        if (stopped)
        {
            text = text.Substring(0, stopAt);
            keptTokens = TokensCoveringPrefix(ctx, genTokens, stopAt);
        }
        string finish = stopped ? "stop" : (genTokens.Count >= sdr.MaxNewTokens ? "length" : "stop");

        ChatLogProbs? logProbs = null;
        if (resp.LogProbs is { } lps)
        {
            int lpCount = Math.Min(keptTokens, lps.Count);
            var content = new List<ChatLogProbContent>(lpCount);
            for (int idx = 0; idx < lpCount; idx++)
            {
                var p = lps[idx];
                content.Add(new ChatLogProbContent
                {
                    Token = DecodeToken(ctx, p.TokenId),
                    LogProb = p.LogProb,
                    TopLogProbs = p.TopLogProbs
                        .Select(t => new ChatTopLogProb { Token = DecodeToken(ctx, t.TokenId), LogProb = t.LogProb })
                        .ToList()
                });
            }
            logProbs = new ChatLogProbs { Content = content };
        }
        return (text, keptTokens, finish, logProbs);
    }

    // Smallest number of leading tokens whose decoded text covers <paramref name="charCount"/> characters —
    // maps a decoded stop-string offset back to a token count so usage/logprobs align with the emitted text.
    private static int TokensCoveringPrefix(GenContext ctx, List<int> tokens, int charCount)
    {
        if (charCount <= 0) return 0;
        // Accumulate per-token decoded lengths once (O(n)) rather than re-decoding every growing prefix
        // (O(n^2) formatting + allocation). Used only to map a decoded stop offset back to a token count.
        var single = new List<int>(1) { 0 };
        int accumulated = 0;
        for (int k = 0; k < tokens.Count; k++)
        {
            single[0] = tokens[k];
            accumulated += ctx.Tokenizer.Decode(single, skipSpecialTokens: true).Length;
            if (accumulated >= charCount) return k + 1;
        }
        return tokens.Count;
    }

    private static string DecodeToken(GenContext ctx, int tokenId)
        => ctx.Tokenizer.Decode(new List<int> { tokenId }, skipSpecialTokens: false) ?? string.Empty;

    /// <summary>Splits an OpenAI <c>model</c> value of the form "baseModel@adapter" into the base model name
    /// and optional multi-LoRA adapter id. Without an "@", the whole string is the base model and there is no
    /// adapter.</summary>
    private static (string BaseModel, string? AdapterId) SplitModelAndAdapter(string model)
    {
        if (string.IsNullOrEmpty(model))
        {
            return (model ?? string.Empty, null);
        }
        int at = model.IndexOf('@');
        if (at < 0)
        {
            return (model, null);
        }
        // Reject a malformed multi-LoRA identifier instead of silently mis-parsing it: at most one '@', and
        // both the base model and the adapter must be non-empty.
        if (model.IndexOf('@', at + 1) >= 0)
        {
            throw new ArgumentException(
                $"Invalid model identifier '{model}': at most one '@' (separating base model and adapter) is allowed.");
        }
        string baseModel = model.Substring(0, at);
        string adapter = model.Substring(at + 1);
        if (string.IsNullOrWhiteSpace(baseModel) || string.IsNullOrWhiteSpace(adapter))
        {
            throw new ArgumentException(
                $"Invalid model identifier '{model}': expected 'baseModel@adapter' with a non-empty base model and adapter.");
        }
        return (baseModel, adapter);
    }

    /// <summary>Parses an OpenAI <c>logit_bias</c> map (token-id string -&gt; bias number) into a typed
    /// dictionary. Throws <see cref="ArgumentException"/> for malformed entries so the caller returns 400.</summary>
    private static IReadOnlyDictionary<int, float>? ParseLogitBias(JObject? logitBias)
    {
        if (logitBias is null || logitBias.Count == 0)
        {
            return null;
        }
        var map = new Dictionary<int, float>(logitBias.Count);
        foreach (var prop in logitBias.Properties())
        {
            if (!int.TryParse(prop.Name, out int tokenId))
            {
                throw new ArgumentException($"'logit_bias' keys must be integer token ids; got '{prop.Name}'.");
            }
            if (prop.Value.Type is not (JTokenType.Integer or JTokenType.Float))
            {
                throw new ArgumentException($"'logit_bias' values must be numbers; got '{prop.Value}' for token {tokenId}.");
            }
            map[tokenId] = prop.Value.Value<float>();
        }
        return map;
    }

    /// <summary>Runs generation to completion, applying stop strings. Returns (text, tokenCount, finishReason).</summary>
    private (string Text, int Count, string Finish) Collect(GenContext ctx, SpeculativeDecodingRequest sdr, IReadOnlyList<string> stops, CancellationToken ct)
    {
        var ids = new List<int>();
        string text = string.Empty;
        bool stopped = false;

        foreach (int tok in _textGeneration.GenerateStream(ctx.ModelName, ctx.NumericType, sdr, ct))
        {
            ids.Add(tok);
            text = ctx.Tokenizer.Decode(ids, skipSpecialTokens: true);
            int stopAt = FindStop(text, stops);
            if (stopAt >= 0)
            {
                text = text.Substring(0, stopAt);
                stopped = true;
                break;
            }
        }

        string finish = stopped ? "stop" : (ids.Count >= sdr.MaxNewTokens ? "length" : "stop");
        return (text, ids.Count, finish);
    }

    /// <summary>Records Prometheus request metrics for a completed streaming generation, deriving TPOT
    /// (inter-token latency) from the total duration minus TTFT over the remaining tokens.</summary>
    private static void RecordStreamMetrics(int promptTokens, int genCount, double totalSeconds, double? ttftSeconds)
    {
        double? tpot = (ttftSeconds is { } t && genCount > 1) ? (totalSeconds - t) / (genCount - 1) : null;
        ServingMetrics.RecordRequest(success: true, promptTokens, genCount, totalSeconds, ttftSeconds, tpot);
    }

    private static int FindStop(string text, IReadOnlyList<string> stops)
    {
        if (stops.Count == 0) return -1;
        int earliest = -1;
        foreach (var s in stops)
        {
            if (string.IsNullOrEmpty(s)) continue;
            int i = text.IndexOf(s, StringComparison.Ordinal);
            if (i >= 0 && (earliest < 0 || i < earliest)) earliest = i;
        }
        return earliest;
    }

    // Length of the longest suffix of <paramref name="text"/> that is a PROPER prefix of some stop string —
    // i.e. text the streamer must hold back because it could still grow into a stop match. Returns 0 when no
    // trailing partial-stop is pending. A completed stop is handled by <see cref="FindStop"/>, not here.
    private static int PendingStopPrefixLen(string text, IReadOnlyList<string> stops)
    {
        int held = 0;
        foreach (var s in stops)
        {
            if (string.IsNullOrEmpty(s)) continue;
            int max = Math.Min(s.Length - 1, text.Length);
            for (int k = max; k > held; k--)
            {
                if (string.CompareOrdinal(text, text.Length - k, s, 0, k) == 0)
                {
                    held = k;
                    break;
                }
            }
        }
        return held;
    }

    private int ResolveEosTokenId(ITokenizer tokenizer)
    {
        var eos = tokenizer.SpecialTokens?.EosToken;
        if (string.IsNullOrEmpty(eos)) return -1;
        try
        {
            var ids = tokenizer.ConvertTokensToIds(new List<string> { eos! });
            return ids.Count > 0 ? ids[0] : -1;
        }
        catch (Exception ex)
        {
            // Log for developer monitoring, then fall back to "no EOS": a tokenizer/config defect must be
            // visible, not silently swallowed (it changes generation termination behavior).
            _logger.LogWarning(ex, "Failed to resolve the EOS token id for the tokenizer; generation will rely on max_tokens/stop only.");
            return -1;
        }
    }

    private static ChatCompletionChunk ContentChunk(string id, long created, string model, string content) => new()
    {
        Id = id, Created = created, Model = model,
        Choices = { new ChatChoice { Index = 0, Delta = new ChatMessageOut { Content = content } } }
    };

    private static CompletionResponse CompletionChunk(string id, long created, string model, string text, string? finish) => new()
    {
        Id = id, Created = created, Model = model,
        Choices = { new CompletionChoice { Index = 0, Text = text, FinishReason = finish } }
    };

    private void BeginSse()
    {
        Response.StatusCode = StatusCodes.Status200OK;
        Response.ContentType = "text/event-stream";
        Response.Headers["Cache-Control"] = "no-cache";
        Response.Headers["X-Accel-Buffering"] = "no";
    }

    private async Task WriteChunkAsync(object chunk, CancellationToken ct)
    {
        string json = JsonConvert.SerializeObject(chunk);
        await Response.WriteAsync($"data: {json}\n\n", ct);
        await Response.Body.FlushAsync(ct);
    }

    private async Task WriteDoneAsync(CancellationToken ct)
    {
        await Response.WriteAsync("data: [DONE]\n\n", ct);
        await Response.Body.FlushAsync(ct);
    }

    private ObjectResult OpenAiError(int status, string message, string type = "invalid_request_error", string? code = null)
        => new(new { error = new { message, type, param = (string?)null, code } }) { StatusCode = status };
}
