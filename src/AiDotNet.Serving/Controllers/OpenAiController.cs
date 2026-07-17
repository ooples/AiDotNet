using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Models.OpenAi;
using AiDotNet.Serving.Services;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Validation;
using Microsoft.AspNetCore.Mvc;
using Newtonsoft.Json;

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

    public OpenAiController(
        IModelRepository modelRepository,
        ITokenizerRegistry tokenizers,
        ITextGenerationService textGeneration,
        ILogger<OpenAiController> logger)
    {
        Guard.NotNull(modelRepository);
        _modelRepository = modelRepository;
        Guard.NotNull(tokenizers);
        _tokenizers = tokenizers;
        Guard.NotNull(textGeneration);
        _textGeneration = textGeneration;
        Guard.NotNull(logger);
        _logger = logger;
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

        if (!TryPrepare(request.Model, out var ctx, out var error))
            return error!;

        string prompt = ChatTemplate.Render(request.Messages.Select(m => (m.Role, m.TextContent())));
        var sdr = BuildRequest(ctx, prompt, request.ResolveMaxTokens(DefaultMaxTokens),
            request.Temperature, request.TopP, request.TopK, request.MinP);
        var stops = request.ResolveStop();
        string id = "chatcmpl-" + Guid.NewGuid().ToString("N");
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        if (!request.Stream)
        {
            var (text, genCount, finish) = Collect(ctx, sdr, stops, ct);
            var response = new ChatCompletionResponse
            {
                Id = id,
                Created = created,
                Model = request.Model,
                Choices = { new ChatChoice { Index = 0, Message = new ChatMessageOut { Role = "assistant", Content = text }, FinishReason = finish } },
                Usage = new Usage { PromptTokens = ctx.PromptTokenCount, CompletionTokens = genCount, TotalTokens = ctx.PromptTokenCount + genCount }
            };
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
        string finishReason = "length";
        bool stopped = false;

        foreach (int tok in _textGeneration.GenerateStream(ctx.ModelName, ctx.NumericType, sdr, ct))
        {
            ids.Add(tok);
            string full = ctx.Tokenizer.Decode(ids, skipSpecialTokens: true);

            int stopAt = FindStop(full, stops);
            if (stopAt >= 0)
            {
                string delta = stopAt > prev.Length ? full.Substring(prev.Length, stopAt - prev.Length) : string.Empty;
                if (delta.Length > 0)
                    await WriteChunkAsync(ContentChunk(id, created, request.Model, delta), ct);
                stopped = true;
                break;
            }

            if (full.Length > prev.Length)
            {
                string delta = full.Substring(prev.Length);
                prev = full;
                await WriteChunkAsync(ContentChunk(id, created, request.Model, delta), ct);
            }
        }

        finishReason = stopped ? "stop" : (ids.Count >= sdr.MaxNewTokens ? "length" : "stop");
        await WriteChunkAsync(new ChatCompletionChunk
        {
            Id = id, Created = created, Model = request.Model,
            Choices = { new ChatChoice { Index = 0, Delta = new ChatMessageOut(), FinishReason = finishReason } }
        }, ct);
        await WriteDoneAsync(ct);
        return new EmptyResult();
    }

    // ============================ /v1/completions ============================

    /// <summary>OpenAI-compatible text completions (streaming and non-streaming).</summary>
    [HttpPost("/v1/completions")]
    public async Task<IActionResult> Completions([FromBody] CompletionRequest request, CancellationToken ct)
    {
        if (request == null || request.Prompt == null)
            return OpenAiError(StatusCodes.Status400BadRequest, "'prompt' is required.");

        if (!TryPrepare(request.Model, out var ctx, out var error))
            return error!;

        var sdr = BuildRequest(ctx, request.PromptText(), request.ResolveMaxTokens(DefaultMaxTokens),
            request.Temperature, request.TopP, request.TopK, request.MinP);
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
            return Ok(response);
        }

        BeginSse();
        var ids = new List<int>();
        string prev = string.Empty;
        bool stopped = false;

        foreach (int tok in _textGeneration.GenerateStream(ctx.ModelName, ctx.NumericType, sdr, ct))
        {
            ids.Add(tok);
            string full = ctx.Tokenizer.Decode(ids, skipSpecialTokens: true);

            int stopAt = FindStop(full, stops);
            if (stopAt >= 0)
            {
                string delta = stopAt > prev.Length ? full.Substring(prev.Length, stopAt - prev.Length) : string.Empty;
                if (delta.Length > 0)
                    await WriteChunkAsync(CompletionChunk(id, created, request.Model, delta, null), ct);
                stopped = true;
                break;
            }

            if (full.Length > prev.Length)
            {
                string delta = full.Substring(prev.Length);
                prev = full;
                await WriteChunkAsync(CompletionChunk(id, created, request.Model, delta, null), ct);
            }
        }

        string finishReason = stopped ? "stop" : (ids.Count >= sdr.MaxNewTokens ? "length" : "stop");
        await WriteChunkAsync(CompletionChunk(id, created, request.Model, string.Empty, finishReason), ct);
        await WriteDoneAsync(ct);
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

    /// <summary>Encodes the prompt and builds the engine request.</summary>
    private static SpeculativeDecodingRequest BuildRequest(GenContext ctx, string prompt, int maxTokens, double? temperature, double? topP, int? topK, double? minP)
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
        };
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

    private static int ResolveEosTokenId(ITokenizer tokenizer)
    {
        var eos = tokenizer.SpecialTokens?.EosToken;
        if (string.IsNullOrEmpty(eos)) return -1;
        try
        {
            var ids = tokenizer.ConvertTokensToIds(new List<string> { eos! });
            return ids.Count > 0 ? ids[0] : -1;
        }
        catch
        {
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
