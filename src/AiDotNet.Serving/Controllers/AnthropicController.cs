using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Models.Anthropic;
using AiDotNet.Serving.Observability;
using AiDotNet.Serving.Services;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Validation;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;

namespace AiDotNet.Serving.Controllers;

/// <summary>
/// Anthropic Messages API surface (<c>POST /v1/messages</c>, <c>POST /v1/messages/count_tokens</c>)
/// so AiDotNet can be driven directly by Claude Code CLI (<c>ANTHROPIC_BASE_URL</c>) and official Anthropic SDK clients.
/// Bridges the text-based Anthropic protocol to the token-ID-native generation engine via the per-model tokenizer.
/// </summary>
[ApiController]
[Produces("application/json")]
public sealed class AnthropicController : ControllerBase
{
    private const int DefaultMaxTokens = 256;

    // Typed null for the anonymous-type SSE payload fields that are null until the message completes.
    private static readonly string? NoStopReason = null;

    private readonly IModelRepository _modelRepository;
    private readonly ITokenizerRegistry _tokenizers;
    private readonly ITextGenerationService _textGeneration;
    private readonly ILogger<AnthropicController> _logger;
    private readonly Configuration.ServingLimitsOptions _limits;

    public AnthropicController(
        IModelRepository modelRepository,
        ITokenizerRegistry tokenizers,
        ITextGenerationService textGeneration,
        ILogger<AnthropicController> logger,
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
        // Same contract as OpenAiController: when no ServingLimitsOptions is registered, the defaults apply,
        // so this route can never bypass the deployment's generation caps.
        _limits = limits ?? new Configuration.ServingLimitsOptions();
    }

    // Enforces the configured per-request generation limits at the HTTP boundary (rejected, not clamped).
    private ObjectResult? CheckLimits(int promptTokens, int requestedMaxTokens)
    {
        if (requestedMaxTokens < 1 || requestedMaxTokens > _limits.EffectiveMaxCompletionTokens)
        {
            return AnthropicError(StatusCodes.Status400BadRequest,
                $"'max_tokens' must be between 1 and {_limits.EffectiveMaxCompletionTokens}.");
        }
        if (promptTokens + requestedMaxTokens > _limits.EffectiveMaxContextTokens)
        {
            return AnthropicError(StatusCodes.Status400BadRequest,
                $"prompt ({promptTokens} tokens) + max_tokens ({requestedMaxTokens}) exceeds the maximum context length of {_limits.EffectiveMaxContextTokens} tokens.");
        }
        return null;
    }

    /// <summary>Anthropic Messages API token counting endpoint (<c>POST /v1/messages/count_tokens</c>).</summary>
    [HttpPost("/v1/messages/count_tokens")]
    [ProducesResponseType(typeof(AnthropicCountTokensResponse), StatusCodes.Status200OK)]
    public IActionResult CountTokens([FromBody] AnthropicMessageRequest? request)
    {
        if (request == null || request.Messages == null || request.Messages.Count == 0)
        {
            return AnthropicError(StatusCodes.Status400BadRequest, "'messages' is required and cannot be empty.");
        }

        string prompt = RenderAnthropicPrompt(request);
        if (TryPrepare(request.Model, out var ctx, out _))
        {
            int tokenCount = Math.Max(1, ctx.Tokenizer.Encode(prompt).TokenIds.Count);
            return Ok(new AnthropicCountTokensResponse { InputTokens = tokenCount });
        }

        // Fallback heuristic if called for token estimation before a model is specified
        int approxTokens = Math.Max(1, (prompt.Length + 3) / 4);
        return Ok(new AnthropicCountTokensResponse { InputTokens = approxTokens });
    }

    /// <summary>Anthropic Messages API chat completion endpoint (<c>POST /v1/messages</c>, streaming and non-streaming).</summary>
    [HttpPost("/v1/messages")]
    public async Task<IActionResult> Messages([FromBody] AnthropicMessageRequest? request, CancellationToken ct)
    {
        if (request == null || request.Messages == null || request.Messages.Count == 0)
        {
            return AnthropicError(StatusCodes.Status400BadRequest, "'messages' is required and cannot be empty.");
        }

        if (!TryPrepare(request.Model, out var ctx, out var error))
        {
            // TryPrepare sets error when it returns false ([NotNullWhen(false)]); the fallback only satisfies analyzers.
            return error ?? AnthropicError(StatusCodes.Status500InternalServerError, "The model could not be prepared.", "api_error");
        }

        string prompt = RenderAnthropicPrompt(request);
        ctx.PromptTokenIds = ctx.Tokenizer.Encode(prompt).TokenIds.ToArray();
        int maxTokens = request.ResolveMaxTokens(DefaultMaxTokens);
        var limitError = CheckLimits(ctx.PromptTokenCount, maxTokens);
        if (limitError is not null)
        {
            return limitError;
        }

        var sdr = new SpeculativeDecodingRequest
        {
            InputTokens = ctx.PromptTokenIds,
            MaxNewTokens = maxTokens,
            Temperature = request.Temperature ?? 1.0,
            TopP = request.TopP ?? 1.0,
            TopK = request.TopK ?? 0,
            EosTokenId = ctx.EosTokenId
        };

        var stops = request.ResolveStop();
        string id = "msg_" + Guid.NewGuid().ToString("N").Substring(0, 24);
        string reportedModel = string.IsNullOrWhiteSpace(request.Model) ? ctx.ModelName : request.Model;
        var sw = Stopwatch.StartNew();

        if (!request.Stream)
        {
            var resp = await _textGeneration.GenerateAsync(ctx.ModelName, ctx.NumericType, sdr, ct).ConfigureAwait(false);
            if (resp.Error is { } err)
            {
                ct.ThrowIfCancellationRequested();
                ServingMetrics.RecordRequest(false, ctx.PromptTokenCount, 0, sw.Elapsed.TotalSeconds);
                // The engine's message can carry internal detail (model names, shapes); log it, return a fixed text,
                // as the streaming path does.
                _logger.LogError("Anthropic generation failed for model {ModelName}: {Error}", ctx.ModelName, err);
                return AnthropicError(StatusCodes.Status500InternalServerError, "Text generation failed.", "api_error");
            }

            var genTokens = new List<int>(resp.GeneratedTokens);
            string text = ctx.Tokenizer.Decode(genTokens, skipSpecialTokens: true);
            int stopAt = StopSequenceMatcher.FindStop(text, stops, out string? matchedStop);
            bool stopped = stopAt >= 0;
            if (stopped)
            {
                text = text.Substring(0, stopAt);
            }

            string stopReason = stopped ? "stop_sequence" : (genTokens.Count >= sdr.MaxNewTokens ? "max_tokens" : "end_turn");
            ServingMetrics.RecordRequest(true, ctx.PromptTokenCount, genTokens.Count, sw.Elapsed.TotalSeconds, null, null);

            return Ok(new AnthropicMessageResponse
            {
                Id = id,
                Model = reportedModel,
                Content = { new AnthropicContentBlock { Type = "text", Text = text } },
                StopReason = stopReason,
                StopSequence = matchedStop,
                Usage = new AnthropicUsage
                {
                    InputTokens = ctx.PromptTokenCount,
                    OutputTokens = genTokens.Count
                }
            });
        }

        // Streaming SSE (Anthropic event-framed SSE protocol)
        BeginSse();
        await WriteAnthropicEventAsync("message_start", new
        {
            type = "message_start",
            message = new
            {
                id,
                type = "message",
                role = "assistant",
                model = reportedModel,
                content = Array.Empty<object>(),
                stop_reason = NoStopReason,
                stop_sequence = NoStopReason,
                usage = new { input_tokens = ctx.PromptTokenCount, output_tokens = 0 }
            }
        }, ct).ConfigureAwait(false);

        await WriteAnthropicEventAsync("content_block_start", new
        {
            type = "content_block_start",
            index = 0,
            content_block = new { type = "text", text = string.Empty }
        }, ct).ConfigureAwait(false);

        await WriteAnthropicEventAsync("ping", new { type = "ping" }, ct).ConfigureAwait(false);

        var ids = new List<int>();
        string prev = string.Empty;
        string lastFull = string.Empty;
        bool streamStopped = false;
        string? streamMatchedStop = null;
        double? ttftSeconds = null;

        try
        {
            foreach (int tok in _textGeneration.GenerateStream(ctx.ModelName, ctx.NumericType, sdr, ct))
            {
                ttftSeconds ??= sw.Elapsed.TotalSeconds;
                ids.Add(tok);
                string full = ctx.Tokenizer.Decode(ids, skipSpecialTokens: true);
                lastFull = full;

                int stopAt = StopSequenceMatcher.FindStop(full, stops, out streamMatchedStop);
                if (stopAt >= 0)
                {
                    string delta = stopAt > prev.Length ? full.Substring(prev.Length, stopAt - prev.Length) : string.Empty;
                    if (delta.Length > 0)
                    {
                        await WriteTextDeltaAsync(delta, ct).ConfigureAwait(false);
                    }
                    prev = full.Substring(0, stopAt);
                    streamStopped = true;
                    break;
                }

                // Hold back any suffix that could still grow into a stop string: streamed text cannot be
                // retracted, so "EN" of stop "END" must not reach the client before "D" settles the match.
                int safeLen = full.Length - StopSequenceMatcher.PendingPrefixLength(full, stops);
                if (safeLen > prev.Length)
                {
                    string delta = full.Substring(prev.Length, safeLen - prev.Length);
                    prev = full.Substring(0, safeLen);
                    await WriteTextDeltaAsync(delta, ct).ConfigureAwait(false);
                }
            }

            // No stop matched: the held-back tail turned out to be ordinary text, so flush it.
            if (!streamStopped && lastFull.Length > prev.Length)
            {
                await WriteTextDeltaAsync(lastFull.Substring(prev.Length), ct).ConfigureAwait(false);
            }
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // message_start is already on the wire, so an HTTP error status is no longer possible. Report the
            // failure in-band as Anthropic's `error` SSE event and end the stream without a normal completion.
            _logger.LogError(ex, "Anthropic streaming generation failed for model {ModelName}", ctx.ModelName);
            ServingMetrics.RecordRequest(false, ctx.PromptTokenCount, ids.Count, sw.Elapsed.TotalSeconds);
            if (!HttpContext.RequestAborted.IsCancellationRequested)
            {
                await WriteAnthropicEventAsync("error", new
                {
                    type = "error",
                    error = new { type = "api_error", message = "Text generation failed." }
                }, ct).ConfigureAwait(false);
            }
            return new EmptyResult();
        }

        string finalStopReason = streamStopped ? "stop_sequence" : (ids.Count >= sdr.MaxNewTokens ? "max_tokens" : "end_turn");

        await WriteAnthropicEventAsync("content_block_stop", new
        {
            type = "content_block_stop",
            index = 0
        }, ct).ConfigureAwait(false);

        await WriteAnthropicEventAsync("message_delta", new
        {
            type = "message_delta",
            delta = new
            {
                stop_reason = finalStopReason,
                stop_sequence = streamMatchedStop
            },
            usage = new
            {
                output_tokens = ids.Count
            }
        }, ct).ConfigureAwait(false);

        await WriteAnthropicEventAsync("message_stop", new
        {
            type = "message_stop"
        }, ct).ConfigureAwait(false);

        double totalSec = sw.Elapsed.TotalSeconds;
        double? tpot = (ttftSeconds is { } t && ids.Count > 1) ? (totalSec - t) / (ids.Count - 1) : null;
        ServingMetrics.RecordRequest(true, ctx.PromptTokenCount, ids.Count, totalSec, ttftSeconds, tpot);
        return new EmptyResult();
    }

    private static string RenderAnthropicPrompt(AnthropicMessageRequest request)
    {
        var turns = new List<(string Role, string Content)>();
        string sys = request.SystemText();
        if (!string.IsNullOrWhiteSpace(sys))
        {
            turns.Add(("system", sys));
        }

        foreach (var msg in request.Messages)
        {
            turns.Add((msg.Role, msg.TextContent()));
        }

        return ChatTemplate.Render(turns);
    }

    private sealed class GenContext
    {
        public required string ModelName { get; init; }
        public required Configuration.NumericType NumericType { get; init; }
        public required ITokenizer Tokenizer { get; init; }
        public int[] PromptTokenIds { get; set; } = Array.Empty<int>();
        public int PromptTokenCount => PromptTokenIds.Length;
        public int? EosTokenId { get; set; }
    }

    private bool TryPrepare(
        string modelName,
        [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out GenContext? ctx,
        [System.Diagnostics.CodeAnalysis.NotNullWhen(false)] out IActionResult? error)
    {
        ctx = null;
        error = null;

        if (string.IsNullOrWhiteSpace(modelName))
        {
            error = AnthropicError(StatusCodes.Status400BadRequest, "'model' is required.");
            return false;
        }

        var info = CliModelAliasResolver.Resolve(modelName, _modelRepository, _textGeneration, _tokenizers, out modelName);
        if (info == null)
        {
            error = AnthropicError(StatusCodes.Status404NotFound, $"The model '{modelName}' does not exist.", "not_found_error");
            return false;
        }

        if (!_textGeneration.SupportsGeneration(modelName, info.NumericType))
        {
            error = AnthropicError(StatusCodes.Status400BadRequest,
                $"The model '{modelName}' does not support text generation.");
            return false;
        }

        if (!_tokenizers.TryGet(modelName, out var tokenizer))
        {
            error = AnthropicError(StatusCodes.Status400BadRequest,
                $"No tokenizer is registered for model '{modelName}'.");
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

    private int ResolveEosTokenId(ITokenizer tokenizer)
    {
        var eos = tokenizer.SpecialTokens?.EosToken;
        if (string.IsNullOrEmpty(eos)) return -1;
        try
        {
            var ids = tokenizer.ConvertTokensToIds(new List<string> { eos });
            return ids.Count > 0 ? ids[0] : -1;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to resolve the EOS token id for the tokenizer.");
            return -1;
        }
    }

    private void BeginSse()
    {
        Response.StatusCode = StatusCodes.Status200OK;
        Response.ContentType = "text/event-stream";
        Response.Headers["Cache-Control"] = "no-cache";
        Response.Headers["X-Accel-Buffering"] = "no";
    }

    private Task WriteTextDeltaAsync(string delta, CancellationToken ct) =>
        WriteAnthropicEventAsync("content_block_delta", new
        {
            type = "content_block_delta",
            index = 0,
            delta = new { type = "text_delta", text = delta }
        }, ct);

    private async Task WriteAnthropicEventAsync(string eventName, object payload, CancellationToken ct)
    {
        string json = JsonConvert.SerializeObject(payload);
        await Response.WriteAsync($"event: {eventName}\ndata: {json}\n\n", ct).ConfigureAwait(false);
        await Response.Body.FlushAsync(ct).ConfigureAwait(false);
    }

    private static ObjectResult AnthropicError(int status, string message, string type = "invalid_request_error")
        => new(new { type = "error", error = new { type, message } }) { StatusCode = status };
}
